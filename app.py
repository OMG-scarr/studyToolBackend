"""
StudyTool - AI-Powered Study Assistant
Main Streamlit application.

A RAG-based learning platform that lets students upload study materials
(PDFs, DOCX, TXT, Markdown, CSV) and query them using natural language.
Powered by LangChain, FAISS, and local sentence-transformer embeddings.
"""

import os
import time
import tempfile
import logging
from pathlib import Path

import streamlit as st
from utils import Config, DocumentProcessor, VectorStoreManager, RAGEngine

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Session State Initialization ---

def init_session_state():
    """Initialize all session state variables."""
    if "processor" not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing" not in st.session_state:
        st.session_state.processing = False


# --- Page Configuration ---

st.set_page_config(
    page_title="StudyTool | AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()


# --- Custom Styling ---

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .source-tag {
        display: inline-block;
        background: #e8f4f8;
        color: #1a5276;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 0.15rem;
    }
    .chat-sources {
        background: #f8f9fa;
        border-left: 3px solid #667eea;
        padding: 0.8rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar: Document Management ---

with st.sidebar:
    st.markdown("### Knowledge Base")

    # Collection stats
    stats = st.session_state.vector_store.get_collection_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{stats["total_chunks"]}</div>'
            f'<div class="stat-label">Chunks</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{stats["unique_sources"]}</div>'
            f'<div class="stat-label">Documents</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # File upload
    st.markdown("#### Upload Study Materials")
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Supports PDF, Word, Text, Markdown, and CSV files.",
    )

    if uploaded_files and st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.processing,
    ):
        st.session_state.processing = True
        progress_bar = st.progress(0)
        status_text = st.empty()

        all_chunks = []
        total = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {uploaded_file.name}...")
            progress_bar.progress((i) / total)

            # Save to temp file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix,
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                chunks = st.session_state.processor.process_file(
                    tmp_path, source_name=uploaded_file.name
                )
                all_chunks.extend(chunks)
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")
            finally:
                os.unlink(tmp_path)

        if all_chunks:
            status_text.text("Embedding documents...")
            progress_bar.progress(0.9)

            added = st.session_state.vector_store.add_documents(all_chunks)

            progress_bar.progress(1.0)
            status_text.text("")
            st.success(f"Added {added} chunks from {total} document(s)")
            time.sleep(1)
            st.rerun()

        st.session_state.processing = False

    st.markdown("---")

    # Document list
    if stats["source_names"]:
        st.markdown("#### Loaded Documents")
        for source in stats["source_names"]:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(f'<span class="source-tag">{source}</span>', unsafe_allow_html=True)
            with col_del:
                if st.button("x", key=f"del_{source}", help=f"Remove {source}"):
                    removed = st.session_state.vector_store.delete_source(source)
                    st.toast(f"Removed {removed} chunks from {source}")
                    st.rerun()

    st.markdown("---")

    # Clear all
    if stats["total_chunks"] > 0:
        if st.button("Clear All Documents", use_container_width=True):
            st.session_state.vector_store.clear_collection()
            st.session_state.chat_history = []
            st.toast("Knowledge base cleared")
            st.rerun()


# --- Main Content: Chat Interface ---

st.markdown('<div class="main-header">StudyTool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Upload your study materials and ask questions. "
    "Powered by semantic search across your documents."
    "</div>",
    unsafe_allow_html=True,
)

# Chat history display
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            source_list = " | ".join(
                [f"**{s['name']}** ({s['relevance']:.0%})" for s in msg["sources"]]
            )
            st.markdown(
                f'<div class="chat-sources">Sources: {source_list}</div>',
                unsafe_allow_html=True,
            )

# Chat input
if stats["total_chunks"] == 0:
    st.info(
        "Upload study materials using the sidebar to get started. "
        "Supports PDF, DOCX, TXT, Markdown, and CSV files."
    )

query = st.chat_input(
    "Ask a question about your study materials...",
    disabled=stats["total_chunks"] == 0,
)

if query:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            response = st.session_state.rag_engine.query(query)

        st.markdown(response.answer)

        if response.sources:
            source_list = " | ".join(
                [f"**{s['name']}** ({s['relevance']:.0%})" for s in response.sources]
            )
            st.markdown(
                f'<div class="chat-sources">Sources: {source_list}</div>',
                unsafe_allow_html=True,
            )

    # Save to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response.answer,
        "sources": response.sources,
    })
