package web

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	"studytool/v2/internal/chunker"
	"studytool/v2/internal/config"
	"studytool/v2/internal/database"
	"studytool/v2/internal/llm"
	"studytool/v2/internal/models"
	"studytool/v2/internal/parser"
	"studytool/v2/internal/vectorstore"
)

type WebHandler struct {
	render *Renderer
	db     *database.DB
	store  *vectorstore.Store
	llm    *llm.Client
	cfg    *config.Config
}

func NewWebHandler(renderer *Renderer, db *database.DB, store *vectorstore.Store, llmClient *llm.Client, cfg *config.Config) *WebHandler {
	return &WebHandler{render: renderer, db: db, store: store, llm: llmClient, cfg: cfg}
}

func (h *WebHandler) Routes(r chi.Router) {
	r.Get("/", h.Home)
	r.Post("/notebooks", h.CreateNotebook)
	r.Get("/notebooks/{notebookID}", h.Notebook)
	r.Put("/notebooks/{notebookID}", h.UpdateNotebook)
	r.Delete("/notebooks/{notebookID}", h.DeleteNotebook)

	r.Post("/notebooks/{notebookID}/sources", h.UploadSource)
	r.Delete("/notebooks/{notebookID}/sources/{sourceID}", h.DeleteSource)

	r.Post("/notebooks/{notebookID}/chat", h.SendChat)
	r.Post("/notebooks/{notebookID}/chat/stream", h.StreamChat)
	r.Delete("/notebooks/{notebookID}/chat/history", h.ClearChat)

	r.Post("/notebooks/{notebookID}/generate", h.Generate)
	r.Delete("/notebooks/{notebookID}/generate/{genID}", h.DeleteGenerated)
}

// --- Pages ---

func (h *WebHandler) Home(w http.ResponseWriter, r *http.Request) {
	notebooks, _ := h.db.ListNotebooks()
	if notebooks == nil {
		notebooks = []models.Notebook{}
	}
	h.render.Render(w, "home", map[string]any{
		"Notebooks": notebooks,
	})
}

func (h *WebHandler) Notebook(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "notebookID")
	nb, err := h.db.GetNotebook(id)
	if err != nil || nb == nil {
		http.Redirect(w, r, "/", http.StatusFound)
		return
	}

	sources, _ := h.db.ListSources(id)
	if sources == nil {
		sources = []models.Source{}
	}
	messages, _ := h.db.GetChatHistory(id, 100)
	if messages == nil {
		messages = []models.ChatMessage{}
	}
	generated, _ := h.db.ListGenerated(id)
	if generated == nil {
		generated = []models.GeneratedContent{}
	}

	h.render.Render(w, "notebook", map[string]any{
		"Notebook":  nb,
		"Sources":   sources,
		"Messages":  messages,
		"Generated": generated,
	})
}

// --- Notebook CRUD ---

func (h *WebHandler) CreateNotebook(w http.ResponseWriter, r *http.Request) {
	title := strings.TrimSpace(r.FormValue("title"))
	if title == "" {
		title = "Untitled Notebook"
	}
	emoji := r.FormValue("emoji")
	if emoji == "" {
		emoji = "📓"
	}

	now := time.Now()
	nb := &models.Notebook{
		ID:        uuid.New().String(),
		Title:     title,
		Emoji:     emoji,
		CreatedAt: now,
		UpdatedAt: now,
	}
	h.db.CreateNotebook(nb)

	// If HTMX request, return just the notebook card
	if r.Header.Get("HX-Request") == "true" {
		w.Header().Set("HX-Redirect", fmt.Sprintf("/notebooks/%s", nb.ID))
		return
	}
	http.Redirect(w, r, fmt.Sprintf("/notebooks/%s", nb.ID), http.StatusFound)
}

func (h *WebHandler) UpdateNotebook(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "notebookID")
	nb, _ := h.db.GetNotebook(id)
	if nb == nil {
		http.Error(w, "not found", 404)
		return
	}

	title := strings.TrimSpace(r.FormValue("title"))
	if title != "" {
		nb.Title = title
	}
	emoji := r.FormValue("emoji")
	if emoji != "" {
		nb.Emoji = emoji
	}
	desc := r.FormValue("description")
	nb.Description = desc

	h.db.UpdateNotebook(nb)

	h.render.RenderPartial(w, "notebook", "notebook_header", map[string]any{"Notebook": nb})
}

func (h *WebHandler) DeleteNotebook(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "notebookID")
	h.store.DeleteByNotebook(id)
	h.store.Save()
	h.db.DeleteNotebook(id)

	w.Header().Set("HX-Redirect", "/")
}

// --- Sources ---

func (h *WebHandler) UploadSource(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	r.ParseMultipartForm(50 << 20)

	file, header, err := r.FormFile("file")
	if err != nil {
		h.renderError(w, "No file provided")
		return
	}
	defer file.Close()

	data, _ := io.ReadAll(file)
	reader := bytes.NewReader(data)
	filename := header.Filename

	text, err := parser.Parse(filename, reader, int64(len(data)))
	if err != nil {
		h.renderError(w, fmt.Sprintf("Failed to parse: %v", err))
		return
	}
	if strings.TrimSpace(text) == "" {
		h.renderError(w, "No text content found")
		return
	}

	chunks := chunker.Chunk(text, h.cfg.ChunkSize, h.cfg.ChunkOverlap)
	if len(chunks) == 0 {
		h.renderError(w, "No content extracted")
		return
	}

	sourceID := uuid.New().String()
	var vectorChunks []vectorstore.Chunk

	batchSize := h.cfg.EmbedBatchSize
	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[i:end]
		embeddings, err := h.llm.EmbedBatch(batch)
		if err != nil {
			h.renderError(w, fmt.Sprintf("Embedding failed: %v", err))
			return
		}
		for j, emb := range embeddings {
			vectorChunks = append(vectorChunks, vectorstore.Chunk{
				ID:         uuid.New().String(),
				SourceID:   sourceID,
				NotebookID: notebookID,
				Content:    batch[j],
				Embedding:  emb,
			})
		}
	}

	h.store.Add(vectorChunks)
	h.store.Save()

	source := &models.Source{
		ID:         sourceID,
		NotebookID: notebookID,
		Filename:   filename,
		FileType:   strings.TrimPrefix(strings.ToLower(filename[strings.LastIndex(filename, "."):]), "."),
		FileSize:   int64(len(data)),
		ChunkCount: len(vectorChunks),
		CreatedAt:  time.Now(),
	}
	h.db.CreateSource(source)
	h.db.TouchNotebook(notebookID)

	log.Printf("[UPLOAD] %s: %d chunks added to notebook", filename, len(vectorChunks))

	// Return updated source list
	sources, _ := h.db.ListSources(notebookID)
	nb, _ := h.db.GetNotebook(notebookID)
	h.render.RenderPartial(w, "notebook", "source_list", map[string]any{
		"Sources":  sources,
		"Notebook": nb,
	})
}

func (h *WebHandler) DeleteSource(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	sourceID := chi.URLParam(r, "sourceID")

	h.store.DeleteBySource(sourceID)
	h.store.Save()
	h.db.DeleteSource(sourceID)
	h.db.TouchNotebook(notebookID)

	sources, _ := h.db.ListSources(notebookID)
	nb, _ := h.db.GetNotebook(notebookID)
	h.render.RenderPartial(w, "notebook", "source_list", map[string]any{
		"Sources":  sources,
		"Notebook": nb,
	})
}

// --- Chat ---

func (h *WebHandler) SendChat(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	message := strings.TrimSpace(r.FormValue("message"))
	if message == "" {
		return
	}

	nb, _ := h.db.GetNotebook(notebookID)
	if nb == nil {
		return
	}

	// Save user message
	userMsg := &models.ChatMessage{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Role:       "user",
		Content:    message,
		CreatedAt:  time.Now(),
	}
	h.db.SaveChatMessage(userMsg)

	// Check for content
	if h.store.CountByNotebook(notebookID) == 0 {
		assistantMsg := &models.ChatMessage{
			ID:         uuid.New().String(),
			NotebookID: notebookID,
			Role:       "assistant",
			Content:    "Please upload some study materials first using the Sources panel on the left.",
			CreatedAt:  time.Now(),
		}
		h.db.SaveChatMessage(assistantMsg)
		h.renderChatMessages(w, notebookID)
		return
	}

	// RAG pipeline
	queryEmb, err := h.llm.Embed(message)
	if err != nil {
		h.saveAndRenderError(w, notebookID, "Failed to process your question. Please try again.")
		return
	}

	results := h.store.Search(queryEmb, h.cfg.SearchTopK, notebookID, nil)
	if len(results) == 0 {
		h.saveAndRenderError(w, notebookID, "I couldn't find relevant information in your documents.")
		return
	}

	var contextParts []string
	var sources []models.ChatSourceRef
	seen := make(map[string]bool)

	for _, res := range results {
		filename := res.Chunk.SourceID
		if src, _ := h.db.GetSource(res.Chunk.SourceID); src != nil {
			filename = src.Filename
		}
		contextParts = append(contextParts, fmt.Sprintf("[From: %s]\n%s", filename, res.Chunk.Content))
		if !seen[res.Chunk.SourceID] {
			seen[res.Chunk.SourceID] = true
			sources = append(sources, models.ChatSourceRef{
				SourceID: res.Chunk.SourceID,
				Filename: filename,
				Relevance: res.Score,
			})
		}
	}

	context := strings.Join(contextParts, "\n\n---\n\n")
	systemPrompt := fmt.Sprintf(`Answer the question using ONLY the context below. Be concise.

Context:
%s`, context)

	history, _ := h.db.GetChatHistory(notebookID, h.cfg.ChatHistoryLen)
	var llmMessages []llm.ChatMessage
	for _, m := range history {
		llmMessages = append(llmMessages, llm.ChatMessage{Role: m.Role, Content: m.Content})
	}
	llmMessages = append(llmMessages, llm.ChatMessage{Role: "user", Content: message})

	answer, err := h.llm.Chat(systemPrompt, llmMessages)
	if err != nil {
		h.saveAndRenderError(w, notebookID, "Generation failed. Is Ollama running?")
		return
	}

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

	h.renderChatMessages(w, notebookID)
}

func (h *WebHandler) StreamChat(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	message := strings.TrimSpace(r.FormValue("message"))
	if message == "" {
		return
	}

	nb, _ := h.db.GetNotebook(notebookID)
	if nb == nil {
		return
	}

	userMsg := &models.ChatMessage{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Role:       "user",
		Content:    message,
		CreatedAt:  time.Now(),
	}
	h.db.SaveChatMessage(userMsg)

	if h.store.CountByNotebook(notebookID) == 0 {
		assistantMsg := &models.ChatMessage{
			ID:         uuid.New().String(),
			NotebookID: notebookID,
			Role:       "assistant",
			Content:    "Please upload some study materials first.",
			CreatedAt:  time.Now(),
		}
		h.db.SaveChatMessage(assistantMsg)
		h.renderChatMessages(w, notebookID)
		return
	}

	queryEmb, err := h.llm.Embed(message)
	if err != nil {
		h.saveAndRenderError(w, notebookID, "Failed to process question.")
		return
	}

	results := h.store.Search(queryEmb, h.cfg.SearchTopK, notebookID, nil)
	var contextParts []string
	var sources []models.ChatSourceRef
	seen := make(map[string]bool)

	for _, res := range results {
		filename := res.Chunk.SourceID
		if src, _ := h.db.GetSource(res.Chunk.SourceID); src != nil {
			filename = src.Filename
		}
		contextParts = append(contextParts, fmt.Sprintf("[From: %s]\n%s", filename, res.Chunk.Content))
		if !seen[res.Chunk.SourceID] {
			seen[res.Chunk.SourceID] = true
			sources = append(sources, models.ChatSourceRef{
				SourceID:  res.Chunk.SourceID,
				Filename:  filename,
				Relevance: res.Score,
			})
		}
	}

	context := strings.Join(contextParts, "\n\n---\n\n")
	systemPrompt := fmt.Sprintf(`Answer the question using ONLY the context below. Be concise.

Context:
%s`, context)

	history, _ := h.db.GetChatHistory(notebookID, h.cfg.ChatHistoryLen)
	var llmMessages []llm.ChatMessage
	for _, m := range history {
		llmMessages = append(llmMessages, llm.ChatMessage{Role: m.Role, Content: m.Content})
	}
	llmMessages = append(llmMessages, llm.ChatMessage{Role: "user", Content: message})

	// SSE streaming
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		h.SendChat(w, r)
		return
	}

	var fullAnswer strings.Builder
	tokenCh := h.llm.ChatStream(systemPrompt, llmMessages)

	for token := range tokenCh {
		if token.Error != nil {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", token.Error.Error())
			flusher.Flush()
			return
		}
		fullAnswer.WriteString(token.Content)
		data, _ := json.Marshal(map[string]any{"token": token.Content, "done": token.Done})
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

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
}

func (h *WebHandler) ClearChat(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	h.db.ClearChatHistory(notebookID)
	h.renderChatMessages(w, notebookID)
}

// --- Generate ---

func (h *WebHandler) Generate(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	genType := r.FormValue("type")

	prompts := map[string]struct {
		title, system, user string
	}{
		"summary": {
			"Summary",
			"You are a study assistant. Summarize concisely.",
			"Summarize the key points using bullet points and bold terms.\n\nMaterial:\n%s",
		},
		"study_guide": {
			"Study Guide",
			"You are a tutor. Create a brief study guide.",
			"Create a study guide: Key Concepts, Important Terms, and 5 Review Questions.\n\nMaterial:\n%s",
		},
		"flashcards": {
			"Flashcards",
			"Create flashcards from study material.",
			"Create 10 flashcards:\n\n**Q:** [Question]\n**A:** [Answer]\n\n---\n\nMaterial:\n%s",
		},
		"faq": {
			"FAQ",
			"Create an FAQ from study material.",
			"Create 5-8 FAQs:\n\n**Q:** [Question]\n**A:** [Answer]\n\nMaterial:\n%s",
		},
		"timeline": {
			"Timeline",
			"Organize information chronologically or logically.",
			"Create a brief timeline or logical progression of the key topics.\n\nMaterial:\n%s",
		},
		"audio_overview": {
			"Audio Overview",
			"Write a short podcast script between Alex and Sam discussing study material.",
			"Write a short conversational script (~500 words) between Alex and Sam covering the key ideas.\n\n**Alex:** [dialogue]\n**Sam:** [dialogue]\n\nMaterial:\n%s",
		},
	}

	prompt, ok := prompts[genType]
	if !ok {
		h.renderError(w, "Invalid generation type")
		return
	}

	chunks := h.store.GetContextForNotebook(notebookID, nil, h.cfg.MaxContextChunks)
	if len(chunks) == 0 {
		h.renderError(w, "Upload some sources first")
		return
	}

	var parts []string
	for _, c := range chunks {
		parts = append(parts, c.Content)
	}
	context := strings.Join(parts, "\n\n---\n\n")
	userPrompt := fmt.Sprintf(prompt.user, context)

	log.Printf("[GENERATE] Creating '%s' for notebook", prompt.title)

	content, err := h.llm.Generate(prompt.system, userPrompt)
	if err != nil {
		h.renderError(w, fmt.Sprintf("Generation failed: %v", err))
		return
	}

	gc := &models.GeneratedContent{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Type:       genType,
		Title:      prompt.title,
		Content:    content,
		CreatedAt:  time.Now(),
	}
	h.db.SaveGenerated(gc)
	h.db.TouchNotebook(notebookID)

	generated, _ := h.db.ListGenerated(notebookID)
	h.render.RenderPartial(w, "notebook", "generated_list", map[string]any{
		"Generated": generated,
		"Notebook":  &models.Notebook{ID: notebookID},
	})
}

func (h *WebHandler) DeleteGenerated(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	genID := chi.URLParam(r, "genID")
	h.db.DeleteGenerated(genID)

	generated, _ := h.db.ListGenerated(notebookID)
	h.render.RenderPartial(w, "notebook", "generated_list", map[string]any{
		"Generated": generated,
		"Notebook":  &models.Notebook{ID: notebookID},
	})
}

// --- Helpers ---

func (h *WebHandler) renderChatMessages(w http.ResponseWriter, notebookID string) {
	messages, _ := h.db.GetChatHistory(notebookID, 100)
	if messages == nil {
		messages = []models.ChatMessage{}
	}
	h.render.RenderPartial(w, "notebook", "chat_messages", map[string]any{
		"Messages": messages,
		"Notebook": &models.Notebook{ID: notebookID},
	})
}

func (h *WebHandler) saveAndRenderError(w http.ResponseWriter, notebookID, msg string) {
	assistantMsg := &models.ChatMessage{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Role:       "assistant",
		Content:    msg,
		CreatedAt:  time.Now(),
	}
	h.db.SaveChatMessage(assistantMsg)
	h.renderChatMessages(w, notebookID)
}

func (h *WebHandler) renderError(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprintf(w, `<div class="toast error">%s</div>`, msg)
}
