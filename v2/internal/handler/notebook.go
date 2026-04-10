package handler

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	"studytool/v2/internal/database"
	"studytool/v2/internal/models"
	"studytool/v2/internal/vectorstore"
)

type NotebookHandler struct {
	db    *database.DB
	store *vectorstore.Store
}

func NewNotebookHandler(db *database.DB, store *vectorstore.Store) *NotebookHandler {
	return &NotebookHandler{db: db, store: store}
}

func (h *NotebookHandler) Create(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Title       string `json:"title"`
		Description string `json:"description"`
		Emoji       string `json:"emoji"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Title == "" {
		jsonError(w, "title is required", http.StatusBadRequest)
		return
	}
	if req.Emoji == "" {
		req.Emoji = "📓"
	}

	now := time.Now()
	nb := &models.Notebook{
		ID:          uuid.New().String(),
		Title:       req.Title,
		Description: req.Description,
		Emoji:       req.Emoji,
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	if err := h.db.CreateNotebook(nb); err != nil {
		jsonError(w, "failed to create notebook", http.StatusInternalServerError)
		return
	}

	jsonResponse(w, http.StatusCreated, nb)
}

func (h *NotebookHandler) List(w http.ResponseWriter, r *http.Request) {
	notebooks, err := h.db.ListNotebooks()
	if err != nil {
		jsonError(w, "failed to list notebooks", http.StatusInternalServerError)
		return
	}
	if notebooks == nil {
		notebooks = []models.Notebook{}
	}
	jsonResponse(w, http.StatusOK, notebooks)
}

func (h *NotebookHandler) Get(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "notebookID")
	nb, err := h.db.GetNotebook(id)
	if err != nil {
		jsonError(w, "failed to get notebook", http.StatusInternalServerError)
		return
	}
	if nb == nil {
		jsonError(w, "notebook not found", http.StatusNotFound)
		return
	}
	jsonResponse(w, http.StatusOK, nb)
}

func (h *NotebookHandler) Update(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "notebookID")
	nb, err := h.db.GetNotebook(id)
	if err != nil || nb == nil {
		jsonError(w, "notebook not found", http.StatusNotFound)
		return
	}

	var req struct {
		Title       *string `json:"title"`
		Description *string `json:"description"`
		Emoji       *string `json:"emoji"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Title != nil {
		nb.Title = *req.Title
	}
	if req.Description != nil {
		nb.Description = *req.Description
	}
	if req.Emoji != nil {
		nb.Emoji = *req.Emoji
	}

	if err := h.db.UpdateNotebook(nb); err != nil {
		jsonError(w, "failed to update notebook", http.StatusInternalServerError)
		return
	}

	jsonResponse(w, http.StatusOK, nb)
}

func (h *NotebookHandler) Delete(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "notebookID")
	nb, err := h.db.GetNotebook(id)
	if err != nil || nb == nil {
		jsonError(w, "notebook not found", http.StatusNotFound)
		return
	}

	h.store.DeleteByNotebook(id)
	h.store.Save()

	if err := h.db.DeleteNotebook(id); err != nil {
		jsonError(w, "failed to delete notebook", http.StatusInternalServerError)
		return
	}

	jsonResponse(w, http.StatusOK, map[string]string{"message": "notebook deleted"})
}
