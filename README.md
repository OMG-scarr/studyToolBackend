# StudyTool - AI-Powered Study Assistant

A RAG-based learning platform that lets students upload study materials and query them using natural language. Think NotebookLM, but open-source and running entirely on your machine.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

## Overview

Students deal with scattered knowledge across PDFs, lecture notes, textbooks, and documentation. StudyTool consolidates everything into a single searchable knowledge base. Upload your materials, ask questions in plain English, and get sourced answers pulled directly from your documents.

No API keys required. All processing runs locally using sentence-transformer embeddings and FAISS vector storage.

## Features

- **Multi-Format Document Ingestion**: Upload PDFs, Word documents, plain text, Markdown, and CSV files
- **Semantic Search**: Vector-based retrieval using sentence-transformers (not keyword matching)
- **Source Attribution**: Every answer cites exactly which document and page the information came from
- **Persistent Knowledge Base**: FAISS stores embeddings on disk across sessions
- **Document Management**: Add, remove, and track individual documents in your collection
- **Chat Interface**: Conversational UI with full message history
- **Zero API Dependency**: Runs entirely on local models with no external service calls

## Architecture

```
User Question
    |
    v
┌──────────────────────────────────────────────────┐
│              Streamlit Frontend                    │
│         (Chat UI + Document Management)           │
└──────────────────┬───────────────────────────────┘
                   |
                   v
┌──────────────────────────────────────────────────┐
│               RAG Engine                          │
│  1. Embed query (sentence-transformers)          │
│  2. Similarity search (FAISS)                    │
│  3. Retrieve top-k relevant chunks               │
│  4. Synthesize sourced answer                    │
└──────────────────┬───────────────────────────────┘
                   |
          ┌────────┴────────┐
          v                 v
┌─────────────────┐  ┌──────────────────┐
│  Document       │  │  FAISS           │
│  Processor      │  │  Vector Store    │
│                 │  │                  │
│  PDF, DOCX,    │  │  Embeddings +    │
│  TXT, MD, CSV  │  │  Metadata        │
│  -> Chunks      │  │  (persistent)    │
└─────────────────┘  └──────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
git clone https://github.com/OMG-scarr/studyToolBackend.git
cd studyToolBackend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload documents via the sidebar and start asking questions.

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Chat interface and document management |
| RAG Pipeline | LangChain | Document loading, splitting, retrieval chain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local vector embedding generation |
| Vector Store | FAISS | Persistent similarity search database |
| Document Parsing | PyPDF, python-docx, Unstructured | Multi-format file processing |

## Project Structure

```
studyToolBackend/
├── app.py                      # Streamlit application (main entry point)
├── utils/
│   ├── __init__.py
│   ├── config.py               # Environment configuration
│   ├── document_processor.py   # File loading and chunking pipeline
│   ├── vector_store.py         # FAISS management and search
│   └── rag_engine.py           # Query processing and answer synthesis
├── .streamlit/
│   └── config.toml             # UI theme configuration
├── requirements.txt
├── .env.example
└── .gitignore
```

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model for embeddings |
| `FAISS_INDEX_DIR` | `./faiss_index` | FAISS index storage location |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum file upload size |

## Extending

**Adding LLM-powered answers:** The current implementation uses extractive retrieval. To add generative answers, integrate an LLM in `rag_engine.py`:
- **Local**: Ollama (Llama 3, Mistral) via `langchain-ollama`
- **Cloud**: OpenAI, Anthropic, or Google via their respective LangChain integrations

**Adding more file types:** Extend `LOADER_MAP` in `document_processor.py` with any LangChain-compatible document loader.

## License

MIT License - see LICENSE for details.

---

**Built for students who want answers, not more tabs.**
