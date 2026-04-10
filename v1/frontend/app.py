"""
StudyTool - AI-Powered Study Assistant
Streamlit frontend that calls the Go microservice backend.
"""

import time
import logging
import subprocess
import sys
import os
from pathlib import Path

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GO_SERVICE_URL = "http://localhost:8080"


# --- Auto-start Go Service ---

def ensure_go_service():
    """Start the Go backend if it's not already running."""
    try:
        requests.get(f"{GO_SERVICE_URL}/api/stats", timeout=2)
        return  # Already running
    except Exception:
        pass

    # Find the Go binary (sibling backend directory)
    base = Path(__file__).parent.parent / "backend"
    if sys.platform == "win32":
        binary = base / "studytool-service.exe"
    else:
        binary = base / "studytool-service"

    if not binary.exists():
        st.error(
            f"Go service binary not found at `{binary}`.\n\n"
            f"Build it first: `cd go-service && go build -o {binary.name} .`"
        )
        st.stop()

    logger.info("Starting Go service...")
    subprocess.Popen(
        [str(binary)],
        cwd=str(base),
    )

    # Wait for it to be ready
    for _ in range(20):
        try:
            time.sleep(0.5)
            requests.get(f"{GO_SERVICE_URL}/api/stats", timeout=2)
            logger.info("Go service is ready")
            return
        except Exception:
            continue

    st.error("Go service failed to start. Check that Ollama is running (`ollama serve`).")
    st.stop()


ensure_go_service()


# --- Helper Functions ---

def get_stats():
    """Fetch collection stats from Go service."""
    try:
        resp = requests.get(f"{GO_SERVICE_URL}/api/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"total_chunks": 0, "unique_sources": 0, "source_names": []}


def upload_file(uploaded_file):
    """Upload a file to the Go service."""
    files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type)}
    resp = requests.post(f"{GO_SERVICE_URL}/api/upload", files=files, timeout=300)
    resp.raise_for_status()
    return resp.json()


def ask_question(question):
    """Send a question to the Go service."""
    resp = requests.post(
        f"{GO_SERVICE_URL}/api/chat",
        json={"question": question},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def delete_source(name):
    """Delete a source from the Go service."""
    resp = requests.delete(f"{GO_SERVICE_URL}/api/source/{name}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def clear_all():
    """Clear all documents from the Go service."""
    resp = requests.delete(f"{GO_SERVICE_URL}/api/clear", timeout=30)
    resp.raise_for_status()
    return resp.json()


# --- Session State ---

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

    stats = get_stats()

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
        total = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {uploaded_file.name}...")
            progress_bar.progress(i / total)

            try:
                result = upload_file(uploaded_file)
                st.success(f"{uploaded_file.name}: {result.get('chunks', 0)} chunks added")
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")

        progress_bar.progress(1.0)
        status_text.text("")
        st.session_state.processing = False
        time.sleep(1)
        st.rerun()

    st.markdown("---")

    # Document list
    if stats.get("source_names"):
        st.markdown("#### Loaded Documents")
        for source in stats["source_names"]:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(f'<span class="source-tag">{source}</span>', unsafe_allow_html=True)
            with col_del:
                if st.button("x", key=f"del_{source}", help=f"Remove {source}"):
                    try:
                        result = delete_source(source)
                        st.toast(f"Removed chunks from {source}")
                    except Exception as e:
                        st.error(f"Failed to remove {source}: {e}")
                    st.rerun()

    st.markdown("---")

    # Clear all
    if stats["total_chunks"] > 0:
        if st.button("Clear All Documents", use_container_width=True):
            clear_all()
            st.session_state.chat_history = []
            st.toast("Knowledge base cleared")
            st.rerun()


# --- Main Content: Chat Interface ---

st.markdown('<div class="main-header">StudyTool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Upload your study materials and ask questions. "
    "Powered by DeepSeek-R1 + Go microservice."
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
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask_question(query)
                answer = response.get("answer", "No response received.")
                sources = response.get("sources", [])
            except Exception as e:
                answer = f"Error connecting to Go service: {e}\n\nMake sure the Go service is running on port 8080."
                sources = []

        st.markdown(answer)

        if sources:
            source_list = " | ".join(
                [f"**{s['name']}** ({s['relevance']:.0%})" for s in sources]
            )
            st.markdown(
                f'<div class="chat-sources">Sources: {source_list}</div>',
                unsafe_allow_html=True,
            )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
