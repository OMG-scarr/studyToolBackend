package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"studytool-service/internal/chunker"
	"studytool-service/internal/ollama"
	"studytool-service/internal/parser"
	"studytool-service/internal/store"
)

type Handler struct {
	ollama *ollama.Client
	store  *store.VectorStore
}

func NewHandler(client *ollama.Client, vs *store.VectorStore) *Handler {
	return &Handler{ollama: client, store: vs}
}

// Upload handles file upload, parsing, chunking, embedding, and storage.
func (h *Handler) Upload(w http.ResponseWriter, r *http.Request) {
	// 50MB max
	r.ParseMultipartForm(50 << 20)

	file, header, err := r.FormFile("file")
	if err != nil {
		jsonError(w, "no file provided", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read entire file into memory for ReaderAt interface
	data, err := io.ReadAll(file)
	if err != nil {
		jsonError(w, "failed to read file", http.StatusInternalServerError)
		return
	}

	reader := bytes.NewReader(data)
	filename := header.Filename

	log.Printf("[UPLOAD] ── Received file: %s (%d bytes)", filename, len(data))

	// Parse document
	log.Printf("[PARSE]  ── Extracting text from %s...", filename)
	text, err := parser.Parse(filename, reader, int64(len(data)))
	if err != nil {
		jsonError(w, fmt.Sprintf("failed to parse %s: %v", filename, err), http.StatusBadRequest)
		return
	}

	if strings.TrimSpace(text) == "" {
		jsonError(w, "no text content found in file", http.StatusBadRequest)
		return
	}
	log.Printf("[PARSE]  ── Extracted %d characters of text", len(text))

	// Chunk text
	log.Printf("[CHUNK]  ── Splitting into chunks (size=1000, overlap=200)...")
	chunks := chunker.Chunk(text, 1000, 200)
	if len(chunks) == 0 {
		jsonError(w, "no chunks produced from file", http.StatusBadRequest)
		return
	}
	log.Printf("[CHUNK]  ── Created %d chunks", len(chunks))

	// Embed chunks in batches
	batchSize := 10
	var docs []store.Document
	totalBatches := (len(chunks) + batchSize - 1) / batchSize

	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[i:end]
		batchNum := (i / batchSize) + 1

		log.Printf("[EMBED]  ── Embedding batch %d/%d (%d chunks)...", batchNum, totalBatches, len(batch))
		embeddings, err := h.ollama.EmbedBatch(batch)
		if err != nil {
			jsonError(w, fmt.Sprintf("embedding failed: %v", err), http.StatusInternalServerError)
			return
		}

		for j, emb := range embeddings {
			docs = append(docs, store.Document{
				Content:   batch[j],
				Source:    filename,
				Embedding: emb,
			})
		}
		log.Printf("[EMBED]  ── Batch %d/%d done", batchNum, totalBatches)
	}

	// Store
	h.store.Add(docs)
	if err := h.store.Save(); err != nil {
		log.Printf("[STORE]  ── Warning: failed to persist: %v", err)
	}

	log.Printf("[DONE]   ── Added %d chunks from %s ✓", len(docs), filename)

	jsonResponse(w, map[string]any{
		"message": fmt.Sprintf("Processed %s: %d chunks added", filename, len(docs)),
		"chunks":  len(docs),
		"source":  filename,
	})
}

// Chat handles question answering with RAG.
func (h *Handler) Chat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Question string `json:"question"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Question == "" {
		jsonError(w, "question is required", http.StatusBadRequest)
		return
	}

	log.Printf("[CHAT]   ── Question: %s", req.Question)

	if h.store.Len() == 0 {
		jsonResponse(w, map[string]any{
			"answer":  "No documents uploaded yet. Please upload some study materials first.",
			"sources": []any{},
		})
		return
	}

	// Embed the question
	log.Printf("[CHAT]   ── Embedding question...")
	queryEmb, err := h.ollama.Embed(req.Question)
	if err != nil {
		jsonError(w, fmt.Sprintf("failed to embed question: %v", err), http.StatusInternalServerError)
		return
	}
	log.Printf("[CHAT]   ── Question embedded")

	// Search for relevant chunks
	log.Printf("[SEARCH] ── Searching %d chunks for relevant context...", h.store.Len())
	results := h.store.Search(queryEmb, 5)

	if len(results) == 0 {
		log.Printf("[SEARCH] ── No relevant chunks found")
		jsonResponse(w, map[string]any{
			"answer":  "I couldn't find relevant information in your documents.",
			"sources": []any{},
		})
		return
	}

	// Build context from results
	var contextParts []string
	seenSources := make(map[string]bool)
	var sources []map[string]any

	for i, res := range results {
		log.Printf("[SEARCH] ── Match %d: score=%.4f source=%s", i+1, res.Score, res.Document.Source)
		contextParts = append(contextParts, fmt.Sprintf("[From: %s]\n%s", res.Document.Source, res.Document.Content))

		if !seenSources[res.Document.Source] {
			seenSources[res.Document.Source] = true
			sources = append(sources, map[string]any{
				"name":      res.Document.Source,
				"relevance": res.Score,
			})
		}
	}

	context := strings.Join(contextParts, "\n\n---\n\n")

	// Generate answer
	log.Printf("[LLM]    ── Generating answer with deepseek-r1...")
	answer, err := h.ollama.Chat(req.Question, context)
	if err != nil {
		jsonError(w, fmt.Sprintf("chat generation failed: %v", err), http.StatusInternalServerError)
		return
	}
	log.Printf("[LLM]    ── Answer generated (%d chars) ✓", len(answer))

	jsonResponse(w, map[string]any{
		"answer":  answer,
		"sources": sources,
	})
}

// Stats returns collection statistics.
func (h *Handler) Stats(w http.ResponseWriter, r *http.Request) {
	jsonResponse(w, h.store.Stats())
}

// Clear removes all documents.
func (h *Handler) Clear(w http.ResponseWriter, r *http.Request) {
	h.store.Clear()
	jsonResponse(w, map[string]string{"message": "Knowledge base cleared"})
}

// DeleteSource removes all chunks from a specific source.
func (h *Handler) DeleteSource(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if name == "" {
		jsonError(w, "source name required", http.StatusBadRequest)
		return
	}

	removed := h.store.DeleteSource(name)
	if err := h.store.Save(); err != nil {
		log.Printf("Warning: failed to persist store: %v", err)
	}

	jsonResponse(w, map[string]any{
		"message": fmt.Sprintf("Removed %d chunks from %s", removed, name),
		"removed": removed,
	})
}

func jsonResponse(w http.ResponseWriter, data any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
