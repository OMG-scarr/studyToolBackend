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

type GenerateHandler struct {
	db    *database.DB
	store *vectorstore.Store
	llm   *llm.Client
}

func NewGenerateHandler(db *database.DB, store *vectorstore.Store, llm *llm.Client) *GenerateHandler {
	return &GenerateHandler{db: db, store: store, llm: llm}
}

var generationPrompts = map[string]struct {
	title  string
	system string
	user   string
}{
	"summary": {
		title:  "Summary",
		system: "You are an expert study assistant. Create a clear, comprehensive summary of the provided study material.",
		user: `Based on the following study material, create a well-organized summary that captures all key concepts, main ideas, and important details.

Use clear markdown formatting with:
- A brief overview paragraph
- Main sections with headers
- Key points as bullet points
- Important terms in **bold**

Study Material:
%s`,
	},
	"study_guide": {
		title:  "Study Guide",
		system: "You are an expert tutor creating a structured study guide.",
		user: `Create a comprehensive study guide from the following material. Structure it as:

## Learning Objectives
- What the student should know after studying

## Key Concepts
- Each major concept explained clearly

## Important Terms & Definitions
- Term: Definition format

## Review Questions
- Questions to test understanding

## Study Tips
- Specific advice for mastering this material

Study Material:
%s`,
	},
	"flashcards": {
		title:  "Flashcards",
		system: "You are an expert at creating effective flashcards for studying. Create flashcards in a clear Q&A format.",
		user: `Create a set of flashcards from the following study material. Format each flashcard as:

**Q:** [Question]
**A:** [Answer]

---

Create 15-25 flashcards covering all key concepts, terms, and important details. Mix factual recall, conceptual understanding, and application questions.

Study Material:
%s`,
	},
	"faq": {
		title:  "FAQ",
		system: "You are a knowledgeable teaching assistant creating an FAQ document.",
		user: `Based on the following study material, create a comprehensive FAQ (Frequently Asked Questions) document.

Include 10-20 questions that students would commonly ask about this material. Cover:
- Basic understanding questions
- "Why" and "How" questions
- Common misconceptions
- Practical application questions

Format each as:
### Q: [Question]
[Detailed answer]

Study Material:
%s`,
	},
	"timeline": {
		title:  "Timeline",
		system: "You are an expert at organizing information chronologically and creating timelines.",
		user: `Create a timeline or logical progression from the following study material. If the material has chronological events, organize them in order. If not, create a logical progression of concepts from foundational to advanced.

Format as:
### [Phase/Period/Step]
- Key events or concepts
- Important details

Study Material:
%s`,
	},
	"audio_overview": {
		title:  "Audio Overview Script",
		system: `You are a podcast script writer. Create an engaging, conversational podcast-style script between two hosts discussing the study material. Make it educational but entertaining, like NotebookLM's Audio Overview feature.

The hosts should:
- Have a natural, conversational tone
- Explain concepts clearly with analogies
- Ask each other questions
- Express genuine curiosity and enthusiasm
- Make complex topics accessible`,
		user: `Create a podcast-style discussion script (about 2000-3000 words) between two hosts, Alex and Sam, discussing the following study material. They should cover all the key concepts in an engaging, conversational way.

Format:
**Alex:** [dialogue]
**Sam:** [dialogue]

Start with a brief intro and end with key takeaways.

Study Material:
%s`,
	},
}

func (h *GenerateHandler) Generate(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")

	nb, err := h.db.GetNotebook(notebookID)
	if err != nil || nb == nil {
		jsonError(w, "notebook not found", http.StatusNotFound)
		return
	}

	var req models.GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}

	prompt, ok := generationPrompts[req.Type]
	if !ok && req.CustomPrompt == "" {
		validTypes := make([]string, 0, len(generationPrompts))
		for k := range generationPrompts {
			validTypes = append(validTypes, k)
		}
		jsonError(w, fmt.Sprintf("invalid type. valid types: %s, or provide custom_prompt", strings.Join(validTypes, ", ")), http.StatusBadRequest)
		return
	}

	// Gather context from notebook
	chunks := h.store.GetContextForNotebook(notebookID, req.SourceIDs, 50)
	if len(chunks) == 0 {
		jsonError(w, "no content available in this notebook", http.StatusBadRequest)
		return
	}

	var contextParts []string
	for _, c := range chunks {
		contextParts = append(contextParts, c.Content)
	}
	context := strings.Join(contextParts, "\n\n---\n\n")

	// Build prompts
	var systemPrompt, userPrompt, title string
	if req.CustomPrompt != "" {
		systemPrompt = "You are a helpful study assistant. Follow the user's instructions carefully based on the provided study material."
		userPrompt = fmt.Sprintf("%s\n\nStudy Material:\n%s", req.CustomPrompt, context)
		title = "Custom Generation"
		if req.Type != "" {
			title = req.Type
		}
	} else {
		systemPrompt = prompt.system
		userPrompt = fmt.Sprintf(prompt.user, context)
		title = prompt.title
	}

	log.Printf("[GENERATE] Creating '%s' for notebook '%s' (%d chunks of context)", title, nb.Title, len(chunks))

	content, err := h.llm.Generate(systemPrompt, userPrompt)
	if err != nil {
		jsonError(w, fmt.Sprintf("generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("[GENERATE] Generated %d chars", len(content))

	gc := &models.GeneratedContent{
		ID:         uuid.New().String(),
		NotebookID: notebookID,
		Type:       req.Type,
		Title:      title,
		Content:    content,
		CreatedAt:  time.Now(),
	}

	if err := h.db.SaveGenerated(gc); err != nil {
		jsonError(w, "failed to save generated content", http.StatusInternalServerError)
		return
	}

	h.db.TouchNotebook(notebookID)

	jsonResponse(w, http.StatusCreated, gc)
}

func (h *GenerateHandler) List(w http.ResponseWriter, r *http.Request) {
	notebookID := chi.URLParam(r, "notebookID")
	items, err := h.db.ListGenerated(notebookID)
	if err != nil {
		jsonError(w, "failed to list generated content", http.StatusInternalServerError)
		return
	}
	if items == nil {
		items = []models.GeneratedContent{}
	}
	jsonResponse(w, http.StatusOK, items)
}

func (h *GenerateHandler) Delete(w http.ResponseWriter, r *http.Request) {
	genID := chi.URLParam(r, "genID")
	if err := h.db.DeleteGenerated(genID); err != nil {
		jsonError(w, "failed to delete", http.StatusInternalServerError)
		return
	}
	jsonResponse(w, http.StatusOK, map[string]string{"message": "deleted"})
}
