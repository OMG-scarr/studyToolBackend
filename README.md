# StudyTool - AI-Powered Study Assistant

A NotebookLM-style learning platform that lets students organize study materials into notebooks, chat with their documents, and generate study aids. Runs entirely on your machine using DeepSeek-R1 via Ollama.

![Go](https://img.shields.io/badge/Go-1.23+-00ADD8?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-DeepSeek--R1-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

No cloud APIs. No API keys. Everything runs locally.

## Features (v2)

- **Multi-Notebook Organization**: Group sources into separate notebooks
- **Multi-Format Upload**: PDF, Word, plain text, Markdown, CSV
- **RAG Chat with Streaming**: SSE token-by-token streaming with source citations
- **Persistent Chat History**: Conversations saved per notebook
- **Content Generation**:
  - Summaries
  - Study Guides
  - Flashcards
  - FAQs
  - Timelines
  - Audio Overview scripts (podcast-style, two hosts)
  - Custom prompts
- **SQLite Persistence**: Notebooks, sources, chat history, generated content
- **Semantic Search**: Per-notebook vector store with cosine similarity
- **Zero Cloud Dependency**: Runs entirely on local models via Ollama

## Quick Start

### Prerequisites

- [Go 1.23+](https://go.dev/dl/)
- [Ollama](https://ollama.com)

### 1. Pull the models

```bash
ollama pull deepseek-r1:1.5b
ollama pull all-minilm
```

### 2. Build and run

```bash
cd v2
go build -o studytool-v2.exe ./cmd/server/
./studytool-v2.exe
```

The API is available at `http://localhost:8081/api/v2/`.

## API

### Notebooks

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v2/notebooks` | Create notebook |
| `GET` | `/api/v2/notebooks` | List notebooks |
| `GET` | `/api/v2/notebooks/{id}` | Get notebook |
| `PUT` | `/api/v2/notebooks/{id}` | Update notebook |
| `DELETE` | `/api/v2/notebooks/{id}` | Delete notebook |

### Sources

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v2/notebooks/{id}/sources` | Upload file (multipart) |
| `GET` | `/api/v2/notebooks/{id}/sources` | List sources |
| `DELETE` | `/api/v2/notebooks/{id}/sources/{sid}` | Delete source |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v2/notebooks/{id}/chat` | Send message (full response) |
| `POST` | `/api/v2/notebooks/{id}/chat/stream` | Send message (SSE streaming) |
| `GET` | `/api/v2/notebooks/{id}/chat/history` | Get chat history |
| `DELETE` | `/api/v2/notebooks/{id}/chat/history` | Clear chat history |

### Generate

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v2/notebooks/{id}/generate` | Generate content |
| `GET` | `/api/v2/notebooks/{id}/generate` | List generated content |
| `DELETE` | `/api/v2/notebooks/{id}/generate/{gid}` | Delete generated content |

**Generation types:** `summary`, `study_guide`, `flashcards`, `faq`, `timeline`, `audio_overview`, or provide `custom_prompt`.

## Architecture (v2)

```
Client (any frontend)
    |  HTTP/SSE
    v
Chi Router (:8081)  ──  Middleware (CORS, Logger, Recovery)
    |
    ├── Notebook Handler  ── SQLite (notebooks, sources, chat, generated)
    ├── Source Handler    ── Parser ── Chunker ── Ollama Embed ── Vector Store
    ├── Chat Handler      ── Vector Search ── Ollama Chat (streaming)
    └── Generate Handler  ── Vector Context ── Ollama Generate
                                |
                                v
                        Ollama (:11434)
                        DeepSeek-R1 + all-minilm
```

## Project Structure

```
studyToolBackend/
├── v1/                              # Version 1 (reference)
│   ├── frontend/                    #   Streamlit UI
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── .streamlit/config.toml
│   └── backend/                     #   Go microservice (stdlib)
│       ├── main.go
│       ├── go.mod
│       └── internal/
│           ├── api/handler.go
│           ├── ollama/client.go
│           ├── store/vector.go
│           ├── parser/parser.go
│           └── chunker/chunker.go
│
├── v2/                              # Version 2 (active)
│   ├── cmd/server/main.go          #   Entry point (Chi router)
│   ├── go.mod
│   └── internal/
│       ├── config/config.go         #   Env-based configuration
│       ├── models/models.go         #   Domain models
│       ├── database/sqlite.go       #   SQLite persistence (pure-Go)
│       ├── vectorstore/store.go     #   Per-notebook vector store
│       ├── llm/ollama.go            #   Ollama client + SSE streaming
│       ├── handler/
│       │   ├── notebook.go          #   Notebook CRUD
│       │   ├── source.go            #   Source upload/manage
│       │   ├── chat.go              #   RAG chat + streaming + history
│       │   ├── generate.go          #   Content generation
│       │   └── helpers.go           #   JSON utilities
│       ├── parser/parser.go         #   Document parsing
│       └── chunker/chunker.go       #   Text chunking
│
├── .gitignore
├── LICENSE
└── README.md
```

## Configuration

Environment variables for v2:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8081` | Server port |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `CHAT_MODEL` | `deepseek-r1:1.5b` | Model for chat/generation |
| `EMBED_MODEL` | `all-minilm` | Model for embeddings |
| `DATA_DIR` | `./data` | SQLite + vector store directory |

## License

MIT License - see LICENSE for details.

---

**Built for students who want answers, not more tabs.**
