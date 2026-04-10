package handler

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	"studytool/v2/internal/chunker"
	"studytool/v2/internal/database"
	"studytool/v2/internal/llm"
	"studytool/v2/internal/models"
	"studytool/v2/internal/parser"
	"studytool/v2/internal/vectorstore"
)

type SourceHandler struct {
	db    *database.DB
	store *vectorstore.Store
	llm   *llm.Client
}

func NewSourceHandler(db *database.DB, store *vectorstore.Store, llm *llm.Client) *SourceHandler {
	return &SourceHandler{db: db, store: store, llm: llm}
}

func (h *SourceHandler) Upload(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")

	// Verify notebook exists
	nb, err := h.db.GetNotebook(notebookID)
	if err != nil || nb == nil {
		jsonError(w, "notebook not found", http.StatusNotFound)
		return
	}

	// 50MB max
	r.ParseMultipartForm(50 << 20)

	file, header, err := r.FormFile("file")
	if err != nil {
		jsonError(w, "no file provided", http.StatusBadRequest)
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		jsonError(w, "failed to read file", http.StatusInternalServerError)
		return
	}

	reader := bytes.NewReader(data)
	filename := header.Filename

	log.Printf("[UPLOAD] Received: %s (%d bytes) for notebook %s", filename, len(data), nb.Title)

	// Parse document
	text, err := parser.Parse(filename, reader, int64(len(data)))
	if err != nil {
		jsonError(w, fmt.Sprintf("failed to parse %s: %v", filename, err), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(text) == "" {
		jsonError(w, "no text content found in file", http.StatusBadRequest)
		return
	}
	log.Printf("[PARSE]  Extracted %d characters", len(text))

	// Chunk text
	chunks := chunker.Chunk(text, 1000, 200)
	if len(chunks) == 0 {
		jsonError(w, "no chunks produced from file", http.StatusBadRequest)
		return
	}
	log.Printf("[CHUNK]  Created %d chunks", len(chunks))

	sourceID := uuid.New().String()

	// Embed chunks in batches
	batchSize := 10
	var vectorChunks []vectorstore.Chunk
	totalBatches := (len(chunks) + batchSize - 1) / batchSize

	for i := 0; i < len(chunks); i += batchSize {
		end := i + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		batch := chunks[i:end]
		batchNum := (i / batchSize) + 1

		log.Printf("[EMBED]  Batch %d/%d (%d chunks)...", batchNum, totalBatches, len(batch))
		embeddings, err := h.llm.EmbedBatch(batch)
		if err != nil {
			jsonError(w, fmt.Sprintf("embedding failed: %v", err), http.StatusInternalServerError)
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

	// Store vectors
	h.store.Add(vectorChunks)
	if err := h.store.Save(); err != nil {
		log.Printf("[STORE]  Warning: failed to persist: %v", err)
	}

	// Save source metadata
	source := &models.Source{
		ID:         sourceID,
		NotebookID: notebookID,
		Filename:   filename,
		FileType:   strings.TrimPrefix(filepath.Ext(filename), "."),
		FileSize:   int64(len(data)),
		ChunkCount: len(vectorChunks),
		CreatedAt:  time.Now(),
	}
	if err := h.db.CreateSource(source); err != nil {
		jsonError(w, "failed to save source metadata", http.StatusInternalServerError)
		return
	}

	h.db.TouchNotebook(notebookID)

	log.Printf("[DONE]   Added %d chunks from %s", len(vectorChunks), filename)

	jsonResponse(w, http.StatusCreated, source)
}

func (h *SourceHandler) List(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	sources, err := h.db.ListSources(notebookID)
	if err != nil {
		jsonError(w, "failed to list sources", http.StatusInternalServerError)
		return
	}
	if sources == nil {
		sources = []models.Source{}
	}
	jsonResponse(w, http.StatusOK, sources)
}

func (h *SourceHandler) Delete(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	sourceID := chi.URLParam(r, "sourceID")

	source, err := h.db.GetSource(sourceID)
	if err != nil || source == nil || source.NotebookID != notebookID {
		jsonError(w, "source not found", http.StatusNotFound)
		return
	}

	removed := h.store.DeleteBySource(sourceID)
	h.store.Save()

	if err := h.db.DeleteSource(sourceID); err != nil {
		jsonError(w, "failed to delete source", http.StatusInternalServerError)
		return
	}

	h.db.TouchNotebook(notebookID)

	jsonResponse(w, http.StatusOK, map[string]any{
		"message": fmt.Sprintf("Removed %d chunks from %s", removed, source.Filename),
		"removed": removed,
	})
}
