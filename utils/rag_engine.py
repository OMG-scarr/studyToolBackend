"""
RAG (Retrieval-Augmented Generation) engine.
Orchestrates the retrieval pipeline: takes a user query,
finds relevant document chunks, and generates a contextual answer.

Uses local models only - no external API keys required.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    context_chunks: List[str]
    query: str


class RAGEngine:
    """
    Core RAG pipeline for the Study Assistant.

    Retrieves relevant document chunks from the vector store
    and synthesizes them into a coherent, sourced answer.
    Runs entirely locally using sentence-transformers for
    semantic search.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def query(
        self,
        question: str,
        k: int = 5,
        score_threshold: float = 0.3,
    ) -> RAGResponse:
        """
        Process a user question through the RAG pipeline.

        Steps:
        1. Embed the question using the same model as documents
        2. Retrieve top-k relevant chunks via similarity search
        3. Compile source citations
        4. Generate a contextual answer from retrieved chunks

        Args:
            question: The user's question.
            k: Number of chunks to retrieve.
            score_threshold: Minimum relevance score (0-1).

        Returns:
            RAGResponse with answer, sources, and context.
        """
        logger.info(f"Processing query: {question[:80]}...")

        # Retrieve relevant chunks with scores
        results_with_scores = self.vector_store.search_with_scores(question, k=k)

        # Filter by relevance threshold
        filtered = [
            (doc, score)
            for doc, score in results_with_scores
            if score >= score_threshold
        ]

        if not filtered:
            return RAGResponse(
                answer="I couldn't find relevant information in the uploaded documents for this question. Try uploading more materials or rephrasing your query.",
                sources=[],
                context_chunks=[],
                query=question,
            )

        # Extract context and sources
        context_chunks = []
        sources = []
        seen_sources = set()

        for doc, score in filtered:
            context_chunks.append(doc.page_content)

            source_name = doc.metadata.get("source", "Unknown")
            if source_name not in seen_sources:
                seen_sources.add(source_name)
                sources.append({
                    "name": source_name,
                    "type": doc.metadata.get("file_type", "unknown"),
                    "relevance": round(score, 3),
                    "page": doc.metadata.get("page", None),
                })

        # Synthesize answer from context
        answer = self._synthesize_answer(question, context_chunks, sources)

        return RAGResponse(
            answer=answer,
            sources=sources,
            context_chunks=context_chunks,
            query=question,
        )

    def _synthesize_answer(
        self,
        question: str,
        context_chunks: List[str],
        sources: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a coherent answer from retrieved context chunks.

        This implementation uses extractive summarization - pulling
        the most relevant passages and presenting them with source
        attribution. For generative answers, integrate an LLM
        (e.g., Ollama, OpenAI, or HuggingFace inference).

        Args:
            question: Original user question.
            context_chunks: Retrieved document passages.
            sources: Source metadata for citations.

        Returns:
            Formatted answer string with citations.
        """
        if not context_chunks:
            return "No relevant context found."

        # Build answer with source-attributed passages
        answer_parts = []
        answer_parts.append(f"Based on your uploaded documents, here is what I found:\n")

        for i, chunk in enumerate(context_chunks[:5]):
            # Clean and truncate chunk for readability
            clean_chunk = chunk.strip()
            if len(clean_chunk) > 500:
                clean_chunk = clean_chunk[:500] + "..."

            answer_parts.append(f"**Passage {i + 1}:**\n{clean_chunk}\n")

        # Add source citations
        if sources:
            answer_parts.append("\n**Sources:**")
            for src in sources:
                page_info = f" (page {src['page']})" if src.get("page") is not None else ""
                relevance = f" [{src['relevance']:.0%} match]"
                answer_parts.append(f"- {src['name']}{page_info}{relevance}")

        return "\n".join(answer_parts)

    def get_related_topics(self, question: str, k: int = 10) -> List[str]:
        """
        Suggest related topics based on the document collection.

        Args:
            question: The user's current question.
            k: Number of related chunks to analyze.

        Returns:
            List of suggested follow-up topics.
        """
        results = self.vector_store.search(question, k=k)

        # Extract unique source names as topic suggestions
        topics = set()
        for doc in results:
            source = doc.metadata.get("source", "")
            if source:
                topics.add(source)

        return sorted(topics)
