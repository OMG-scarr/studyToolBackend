package store

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// Document is a chunk of text with its embedding and metadata.
type Document struct {
	ID        int
	Content   string
	Source    string
	Embedding []float64
}

// SearchResult holds a document and its similarity score.
type SearchResult struct {
	Document Document
	Score    float64
}

// VectorStore holds documents with their embeddings in memory.
type VectorStore struct {
	mu       sync.RWMutex
	docs     []Document
	nextID   int
	savePath string
}

// New creates a new VectorStore that persists to the given path.
func New(path string) *VectorStore {
	return &VectorStore{savePath: path}
}

// Add inserts documents into the store.
func (vs *VectorStore) Add(docs []Document) {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	for i := range docs {
		docs[i].ID = vs.nextID
		vs.nextID++
	}
	vs.docs = append(vs.docs, docs...)
}

// Search finds the k most similar documents to the query embedding.
func (vs *VectorStore) Search(queryEmbedding []float64, k int) []SearchResult {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	if len(vs.docs) == 0 {
		return nil
	}

	results := make([]SearchResult, 0, len(vs.docs))
	for _, doc := range vs.docs {
		score := cosineSimilarity(queryEmbedding, doc.Embedding)
		results = append(results, SearchResult{Document: doc, Score: score})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if k > len(results) {
		k = len(results)
	}
	return results[:k]
}

// Stats returns collection statistics.
func (vs *VectorStore) Stats() map[string]any {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	sources := make(map[string]bool)
	for _, doc := range vs.docs {
		sources[doc.Source] = true
	}

	sourceNames := make([]string, 0, len(sources))
	for s := range sources {
		sourceNames = append(sourceNames, s)
	}
	sort.Strings(sourceNames)

	return map[string]any{
		"total_chunks":   len(vs.docs),
		"unique_sources": len(sources),
		"source_names":   sourceNames,
	}
}

// DeleteSource removes all documents from a given source.
func (vs *VectorStore) DeleteSource(source string) int {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	var kept []Document
	removed := 0
	for _, doc := range vs.docs {
		if doc.Source == source {
			removed++
		} else {
			kept = append(kept, doc)
		}
	}
	vs.docs = kept
	return removed
}

// Clear removes all documents.
func (vs *VectorStore) Clear() {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.docs = nil
	vs.nextID = 0
	os.Remove(vs.savePath)
}

// Len returns the number of documents.
func (vs *VectorStore) Len() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.docs)
}

// Save persists the store to disk.
func (vs *VectorStore) Save() error {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	dir := filepath.Dir(vs.savePath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create dir: %w", err)
	}

	f, err := os.Create(vs.savePath)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	data := struct {
		Docs   []Document
		NextID int
	}{vs.docs, vs.nextID}

	if err := gob.NewEncoder(f).Encode(data); err != nil {
		return fmt.Errorf("encode: %w", err)
	}

	return nil
}

// Load restores the store from disk.
func (vs *VectorStore) Load() error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	f, err := os.Open(vs.savePath)
	if err != nil {
		return err
	}
	defer f.Close()

	var data struct {
		Docs   []Document
		NextID int
	}
	if err := gob.NewDecoder(f).Decode(&data); err != nil {
		return fmt.Errorf("decode: %w", err)
	}

	vs.docs = data.Docs
	vs.nextID = data.NextID
	return nil
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
