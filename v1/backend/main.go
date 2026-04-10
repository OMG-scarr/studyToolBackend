package main

import (
	"log"
	"net/http"
	"os"
	"studytool-service/internal/api"
	"studytool-service/internal/ollama"
	"studytool-service/internal/store"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	ollamaURL := os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}

	chatModel := os.Getenv("CHAT_MODEL")
	if chatModel == "" {
		chatModel = "deepseek-r1:1.5b"
	}

	embedModel := os.Getenv("EMBED_MODEL")
	if embedModel == "" {
		embedModel = "all-minilm"
	}

	client := ollama.NewClient(ollamaURL, chatModel, embedModel)
	vectorStore := store.New("./data/vectors.gob")

	// Load persisted vectors on startup
	if err := vectorStore.Load(); err != nil {
		log.Printf("No existing vector store found, starting fresh: %v", err)
	} else {
		log.Printf("Loaded %d chunks from disk", vectorStore.Len())
	}

	handler := api.NewHandler(client, vectorStore)
	mux := http.NewServeMux()

	mux.HandleFunc("POST /api/upload", handler.Upload)
	mux.HandleFunc("POST /api/chat", handler.Chat)
	mux.HandleFunc("GET /api/stats", handler.Stats)
	mux.HandleFunc("DELETE /api/clear", handler.Clear)
	mux.HandleFunc("DELETE /api/source/{name}", handler.DeleteSource)

	// CORS middleware for Streamlit
	wrapped := corsMiddleware(mux)

	log.Printf("StudyTool Go service starting on :%s", port)
	log.Printf("Using Ollama at %s (chat: %s, embed: %s)", ollamaURL, chatModel, embedModel)
	log.Fatal(http.ListenAndServe(":"+port, wrapped))
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}
