package config

import (
	"os"
	"strconv"
)

type Config struct {
	Port       string
	OllamaURL  string
	ChatModel  string
	EmbedModel string
	DataDir    string

	// Tuning for hardware constraints
	ChunkSize       int // characters per chunk
	ChunkOverlap    int // overlap between chunks
	EmbedBatchSize  int // chunks per embedding request
	SearchTopK      int // chunks returned from vector search
	MaxContextChunks int // max chunks sent to LLM for generation
	ChatHistoryLen  int // recent messages sent to LLM for context
}

func Load() *Config {
	return &Config{
		Port:       envOr("PORT", "8081"),
		OllamaURL:  envOr("OLLAMA_URL", "http://localhost:11434"),
		ChatModel:  envOr("CHAT_MODEL", "llama3.2:1b"),
		EmbedModel: envOr("EMBED_MODEL", "all-minilm"),
		DataDir:    envOr("DATA_DIR", "./data"),

		ChunkSize:       intEnvOr("CHUNK_SIZE", 500),
		ChunkOverlap:    intEnvOr("CHUNK_OVERLAP", 100),
		EmbedBatchSize:  intEnvOr("EMBED_BATCH_SIZE", 5),
		SearchTopK:      intEnvOr("SEARCH_TOP_K", 3),
		MaxContextChunks: intEnvOr("MAX_CONTEXT_CHUNKS", 15),
		ChatHistoryLen:  intEnvOr("CHAT_HISTORY_LEN", 4),
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func intEnvOr(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}
