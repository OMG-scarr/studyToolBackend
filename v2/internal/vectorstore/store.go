package vectorstore

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// Chunk is a text fragment with its embedding.
type Chunk struct {
	ID         string
	SourceID   string
	NotebookID string
	Content    string
	Embedding  []float64
}

// SearchResult holds a chunk and its similarity score.
type SearchResult struct {
	Chunk Chunk
	Score float64
}

// Store manages vector embeddings with per-notebook isolation.
type Store struct {
	mu       sync.RWMutex
	chunks   []Chunk
	dataDir  string
}

func New(dataDir string) *Store {
	os.MkdirAll(dataDir, 0o755)
	return &Store{dataDir: dataDir}
}

func (s *Store) savePath() string {
	return filepath.Join(s.dataDir, "vectors.gob")
}

// Add inserts chunks into the store.
func (s *Store) Add(chunks []Chunk) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.chunks = append(s.chunks, chunks...)
}

// Search finds the k most similar chunks, optionally filtered by notebook and source IDs.
func (s *Store) Search(queryEmbedding []float64, k int, notebookID string, sourceIDs []string) []SearchResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	sourceSet := make(map[string]bool, len(sourceIDs))
	for _, id := range sourceIDs {
		sourceSet[id] = true
	}

	var results []SearchResult
	for _, chunk := range s.chunks {
		if notebookID != "" && chunk.NotebookID != notebookID {
			continue
		}
		if len(sourceSet) > 0 && !sourceSet[chunk.SourceID] {
			continue
		}
		score := cosineSimilarity(queryEmbedding, chunk.Embedding)
		results = append(results, SearchResult{Chunk: chunk, Score: score})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if k > len(results) {
		k = len(results)
	}
	return results[:k]
}

// DeleteBySource removes all chunks for a given source ID.
func (s *Store) DeleteBySource(sourceID string) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	var kept []Chunk
	removed := 0
	for _, c := range s.chunks {
		if c.SourceID == sourceID {
			removed++
		} else {
			kept = append(kept, c)
		}
	}
	s.chunks = kept
	return removed
}

// DeleteByNotebook removes all chunks for a given notebook ID.
func (s *Store) DeleteByNotebook(notebookID string) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	var kept []Chunk
	removed := 0
	for _, c := range s.chunks {
		if c.NotebookID == notebookID {
			removed++
		} else {
			kept = append(kept, c)
		}
	}
	s.chunks = kept
	return removed
}

// CountByNotebook returns the number of chunks for a notebook.
func (s *Store) CountByNotebook(notebookID string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	count := 0
	for _, c := range s.chunks {
		if c.NotebookID == notebookID {
			count++
		}
	}
	return count
}

// Len returns total chunk count.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.chunks)
}

// Save persists the store to disk.
func (s *Store) Save() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	f, err := os.Create(s.savePath())
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	if err := gob.NewEncoder(f).Encode(s.chunks); err != nil {
		return fmt.Errorf("encode: %w", err)
	}
	return nil
}

// Load restores the store from disk.
func (s *Store) Load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	f, err := os.Open(s.savePath())
	if err != nil {
		return err
	}
	defer f.Close()

	var chunks []Chunk
	if err := gob.NewDecoder(f).Decode(&chunks); err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	s.chunks = chunks
	return nil
}

// GetContextForNotebook returns all chunk content for a notebook, for generation tasks.
func (s *Store) GetContextForNotebook(notebookID string, sourceIDs []string, maxChunks int) []Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()

	sourceSet := make(map[string]bool, len(sourceIDs))
	for _, id := range sourceIDs {
		sourceSet[id] = true
	}

	var result []Chunk
	for _, c := range s.chunks {
		if c.NotebookID != notebookID {
			continue
		}
		if len(sourceSet) > 0 && !sourceSet[c.SourceID] {
			continue
		}
		result = append(result, c)
		if maxChunks > 0 && len(result) >= maxChunks {
			break
		}
	}
	return result
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
