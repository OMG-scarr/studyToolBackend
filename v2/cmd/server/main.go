package main

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/go-chi/chi/v5"
	chimw "github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"

	"studytool/v2/internal/config"
	"studytool/v2/internal/database"
	"studytool/v2/internal/handler"
	"studytool/v2/internal/llm"
	"studytool/v2/internal/vectorstore"
	"studytool/v2/internal/web"
)

func main() {
	cfg := config.Load()

	// Initialize database
	db, err := database.New(cfg.DataDir)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer db.Close()

	// Initialize vector store
	store := vectorstore.New(filepath.Join(cfg.DataDir, "vectors"))
	if err := store.Load(); err != nil {
		log.Printf("No existing vector store found, starting fresh: %v", err)
	} else {
		log.Printf("Loaded %d chunks from disk", store.Len())
	}

	// Initialize LLM client
	llmClient := llm.NewClient(cfg.OllamaURL, cfg.ChatModel, cfg.EmbedModel)

	// Initialize API handlers
	notebookH := handler.NewNotebookHandler(db, store)
	sourceH := handler.NewSourceHandler(db, store, llmClient)
	chatH := handler.NewChatHandler(db, store, llmClient)
	generateH := handler.NewGenerateHandler(db, store, llmClient)

	// Initialize web UI
	templateDir := findWebDir("web/templates")
	staticDir := findWebDir("web/static")
	renderer := web.NewRenderer(templateDir, true)
	webH := web.NewWebHandler(renderer, db, store, llmClient, cfg)

	// Build router
	r := chi.NewRouter()

	// Middleware
	r.Use(chimw.Logger)
	r.Use(chimw.Recoverer)
	r.Use(chimw.RequestID)
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Content-Type", "Authorization"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	// Static files
	r.Handle("/static/*", http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir))))

	// Health check
	r.Get("/api/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status": "ok", "version": "2.0"}`))
	})

	// JSON API routes
	r.Route("/api/v2", func(r chi.Router) {
		r.Route("/notebooks", func(r chi.Router) {
			r.Get("/", notebookH.List)
			r.Post("/", notebookH.Create)

			r.Route("/{notebookID}", func(r chi.Router) {
				r.Get("/", notebookH.Get)
				r.Put("/", notebookH.Update)
				r.Delete("/", notebookH.Delete)

				r.Route("/sources", func(r chi.Router) {
					r.Get("/", sourceH.List)
					r.Post("/", sourceH.Upload)
					r.Delete("/{sourceID}", sourceH.Delete)
				})

				r.Route("/chat", func(r chi.Router) {
					r.Post("/", chatH.Send)
					r.Post("/stream", chatH.Stream)
					r.Get("/history", chatH.History)
					r.Delete("/history", chatH.ClearHistory)
				})

				r.Route("/generate", func(r chi.Router) {
					r.Post("/", generateH.Generate)
					r.Get("/", generateH.List)
					r.Delete("/{genID}", generateH.Delete)
				})
			})
		})
	})

	// Web UI routes (HTML + HTMX)
	r.Group(webH.Routes)

	log.Printf("StudyTool v2 starting on :%s", cfg.Port)
	log.Printf("Using Ollama at %s (chat: %s, embed: %s)", cfg.OllamaURL, cfg.ChatModel, cfg.EmbedModel)
	log.Printf("UI:  http://localhost:%s/", cfg.Port)
	log.Printf("API: http://localhost:%s/api/v2/", cfg.Port)
	log.Fatal(http.ListenAndServe(":"+cfg.Port, r))
}

// findWebDir locates the web directory relative to the executable or working directory.
func findWebDir(rel string) string {
	// Check relative to working directory
	if info, err := os.Stat(rel); err == nil && info.IsDir() {
		abs, _ := filepath.Abs(rel)
		return abs
	}

	// Check relative to executable
	exe, _ := os.Executable()
	dir := filepath.Dir(exe)
	candidate := filepath.Join(dir, rel)
	if info, err := os.Stat(candidate); err == nil && info.IsDir() {
		return candidate
	}

	// Fallback
	abs, _ := filepath.Abs(rel)
	return abs
}
