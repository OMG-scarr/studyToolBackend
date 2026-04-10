package models

import "time"

// Notebook is the top-level organizational unit (like a NotebookLM notebook).
type Notebook struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Emoji       string    `json:"emoji"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	SourceCount int       `json:"source_count,omitempty"`
	ChunkCount  int       `json:"chunk_count,omitempty"`
}

// Source is an uploaded document belonging to a notebook.
type Source struct {
	ID         string    `json:"id"`
	NotebookID string    `json:"notebook_id"`
	Filename   string    `json:"filename"`
	FileType   string    `json:"file_type"`
	FileSize   int64     `json:"file_size"`
	ChunkCount int       `json:"chunk_count"`
	Summary    string    `json:"summary,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
}

// ChatMessage is a single message in a notebook's conversation.
type ChatMessage struct {
	ID         string            `json:"id"`
	NotebookID string            `json:"notebook_id"`
	Role       string            `json:"role"` // "user" or "assistant"
	Content    string            `json:"content"`
	Sources    []ChatSourceRef   `json:"sources,omitempty"`
	CreatedAt  time.Time         `json:"created_at"`
}

// ChatSourceRef references a source document used in a response.
type ChatSourceRef struct {
	SourceID  string  `json:"source_id"`
	Filename  string  `json:"filename"`
	Relevance float64 `json:"relevance"`
	Excerpt   string  `json:"excerpt,omitempty"`
}

// GeneratedContent holds generated study materials.
type GeneratedContent struct {
	ID         string    `json:"id"`
	NotebookID string    `json:"notebook_id"`
	Type       string    `json:"type"` // "summary", "study_guide", "flashcards", "faq", "timeline", "audio_overview"
	Title      string    `json:"title"`
	Content    string    `json:"content"`
	CreatedAt  time.Time `json:"created_at"`
}

// GenerateRequest is the request body for content generation.
type GenerateRequest struct {
	Type        string   `json:"type"`
	CustomPrompt string  `json:"custom_prompt,omitempty"`
	SourceIDs   []string `json:"source_ids,omitempty"` // empty = use all sources
}

// ChatRequest is the request body for chat.
type ChatRequest struct {
	Message string `json:"message"`
}

// Chunk is an embedded text fragment stored in the vector store.
type Chunk struct {
	ID         string
	SourceID   string
	NotebookID string
	Content    string
	Embedding  []float64
}
