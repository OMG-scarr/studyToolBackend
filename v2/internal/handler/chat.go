package handler

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	"studytool/v2/internal/database"
	"studytool/v2/internal/llm"
	"studytool/v2/internal/models"
	"studytool/v2/internal/vectorstore"
)

type ChatHandler struct {
	db    *database.DB
	store *vectorstore.Store
	llm   *llm.Client
}

func NewChatHandler(db *database.DB, store *vectorstore.Store, llm *llm.Client) *ChatHandler {
	return &ChatHandler{db: db, store: store, llm: llm}
}

const ragSystemPrompt = `Answer the question using ONLY the context below. Be concise.

Context:
%s`

// Send handles a chat message (non-streaming with full response).
func (h *ChatHandler) Send(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")

	nb, err := h.db.GetNotebook(notebookID)
	if err != nil || nb == nil {
		jsonError(w, "notebook not found", http.StatusNotFound)
		return
	}

	var req models.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Message == "" {
		jsonError(w, "message is required", http.StatusBadRequest)
		return
	}

	log.Printf("[CHAT] Question in '%s': %s", nb.Title, req.Message)

	// Save user message
	userMsg := &models.ChatMessage{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Role:       "user",
		Content:    req.Message,
		CreatedAt:  time.Now(),
	}
	h.db.SaveChatMessage(userMsg)

	// Check if notebook has content
	chunkCount := h.store.CountByNotebook(notebookID)
	if chunkCount == 0 {
		assistantMsg := &models.ChatMessage{
			ID:         uuid.New().String(),
			NotebookID: notebookID,
			Role:       "assistant",
			Content:    "No documents uploaded yet. Please upload some study materials first using the sources panel.",
			CreatedAt:  time.Now(),
		}
		h.db.SaveChatMessage(assistantMsg)
		jsonResponse(w, http.StatusOK, assistantMsg)
		return
	}

	// Embed the question
	queryEmb, err := h.llm.Embed(req.Message)
	if err != nil {
		jsonError(w, fmt.Sprintf("failed to embed question: %v", err), http.StatusInternalServerError)
		return
	}

	// Search for relevant chunks
	results := h.store.Search(queryEmb, 5, notebookID, nil)
	if len(results) == 0 {
		assistantMsg := &models.ChatMessage{
			ID:         uuid.New().String(),
			NotebookID: notebookID,
			Role:       "assistant",
			Content:    "I couldn't find relevant information in your documents for that question.",
			CreatedAt:  time.Now(),
		}
		h.db.SaveChatMessage(assistantMsg)
		jsonResponse(w, http.StatusOK, assistantMsg)
		return
	}

	// Build context and source references
	context, sources := buildContextAndSources(results, h.db)
	systemPrompt := fmt.Sprintf(ragSystemPrompt, context)

	// Get recent chat history for conversational context
	history, _ := h.db.GetChatHistory(notebookID, 10)
	messages := buildLLMMessages(history, req.Message)

	// Generate answer
	log.Printf("[LLM] Generating answer...")
	answer, err := h.llm.Chat(systemPrompt, messages)
	if err != nil {
		jsonError(w, fmt.Sprintf("generation failed: %v", err), http.StatusInternalServerError)
		return
	}
	log.Printf("[LLM] Answer generated (%d chars)", len(answer))

	// Save assistant message
	assistantMsg := &models.ChatMessage{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Role:       "assistant",
		Content:    answer,
		Sources:    sources,
		CreatedAt:  time.Now(),
	}
	h.db.SaveChatMessage(assistantMsg)
	h.db.TouchNotebook(notebookID)

	jsonResponse(w, http.StatusOK, assistantMsg)
}

// Stream handles a chat message with SSE streaming.
func (h *ChatHandler) Stream(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")

	nb, err := h.db.GetNotebook(notebookID)
	if err != nil || nb == nil {
		jsonError(w, "notebook not found", http.StatusNotFound)
		return
	}

	var req models.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Message == "" {
		jsonError(w, "message is required", http.StatusBadRequest)
		return
	}

	// Save user message
	userMsg := &models.ChatMessage{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Role:       "user",
		Content:    req.Message,
		CreatedAt:  time.Now(),
	}
	h.db.SaveChatMessage(userMsg)

	chunkCount := h.store.CountByNotebook(notebookID)
	if chunkCount == 0 {
		jsonError(w, "no documents uploaded yet", http.StatusBadRequest)
		return
	}

	// Embed and search
	queryEmb, err := h.llm.Embed(req.Message)
	if err != nil {
		jsonError(w, fmt.Sprintf("failed to embed question: %v", err), http.StatusInternalServerError)
		return
	}

	results := h.store.Search(queryEmb, 5, notebookID, nil)
	context, sources := buildContextAndSources(results, h.db)
	systemPrompt := fmt.Sprintf(ragSystemPrompt, context)

	history, _ := h.db.GetChatHistory(notebookID, 10)
	messages := buildLLMMessages(history, req.Message)

	// Set up SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		jsonError(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// Send sources first
	sourcesJSON, _ := json.Marshal(sources)
	fmt.Fprintf(w, "event: sources\ndata: %s\n\n", sourcesJSON)
	flusher.Flush()

	// Stream tokens
	var fullAnswer strings.Builder
	tokenCh := h.llm.ChatStream(systemPrompt, messages)

	for token := range tokenCh {
		if token.Error != nil {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", token.Error.Error())
			flusher.Flush()
			return
		}

		fullAnswer.WriteString(token.Content)
		tokenJSON, _ := json.Marshal(map[string]any{
			"content": token.Content,
			"done":    token.Done,
		})
		fmt.Fprintf(w, "data: %s\n\n", tokenJSON)
		flusher.Flush()
	}

	// Save the complete assistant message
	assistantMsg := &models.ChatMessage{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Role:       "assistant",
		Content:    fullAnswer.String(),
		Sources:    sources,
		CreatedAt:  time.Now(),
	}
	h.db.SaveChatMessage(assistantMsg)
	h.db.TouchNotebook(notebookID)

	// Send done event
	fmt.Fprintf(w, "event: done\ndata: {\"message_id\": \"%s\"}\n\n", assistantMsg.ID)
	flusher.Flush()
}

// History returns chat history for a notebook.
func (h *ChatHandler) History(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	messages, err := h.db.GetChatHistory(notebookID, 100)
	if err != nil {
		jsonError(w, "failed to get chat history", http.StatusInternalServerError)
		return
	}
	if messages == nil {
		messages = []models.ChatMessage{}
	}
	jsonResponse(w, http.StatusOK, messages)
}

// ClearHistory deletes all chat messages for a notebook.
func (h *ChatHandler) ClearHistory(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	if err := h.db.ClearChatHistory(notebookID); err != nil {
		jsonError(w, "failed to clear history", http.StatusInternalServerError)
		return
	}
	jsonResponse(w, http.StatusOK, map[string]string{"message": "chat history cleared"})
}

func buildContextAndSources(results []vectorstore.SearchResult, db *database.DB) (string, []models.ChatSourceRef) {
	var contextParts []string
	seen := make(map[string]bool)
	var sources []models.ChatSourceRef

	for _, res := range results {
		// Look up source filename
		filename := res.Chunk.SourceID
		if src, err := db.GetSource(res.Chunk.SourceID); err == nil && src != nil {
			filename = src.Filename
		}

		contextParts = append(contextParts, fmt.Sprintf("[From: %s]\n%s", filename, res.Chunk.Content))

		if !seen[res.Chunk.SourceID] {
			seen[res.Chunk.SourceID] = true
			sources = append(sources, models.ChatSourceRef{
				SourceID:  res.Chunk.SourceID,
				Filename:  filename,
				Relevance: res.Score,
				Excerpt:   truncate(res.Chunk.Content, 150),
			})
		}
	}

	return strings.Join(contextParts, "\n\n---\n\n"), sources
}

func buildLLMMessages(history []models.ChatMessage, currentMessage string) []llm.ChatMessage {
	var messages []llm.ChatMessage

	// Include recent history for context (skip last user message since we add it fresh)
	for _, msg := range history {
		messages = append(messages, llm.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Current question (already saved to history, but include explicitly)
	messages = append(messages, llm.ChatMessage{
		Role:    "user",
		Content: currentMessage,
	})

	return messages
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
