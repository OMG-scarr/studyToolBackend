"""
Vector store management using FAISS.
Handles document embedding, storage, and similarity search
with persistent local storage.
"""

import json
import logging
import os
import shutil
from typing import List, Optional, Dict, Any

import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils.config import Config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages the FAISS vector store for document embeddings.

    Uses HuggingFace sentence-transformers for local embedding
    generation (no API keys required). Persists data to disk
    so knowledge survives between sessions.
    """

    def __init__(self):
        self._embeddings = None
        self._vector_store = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load the embedding model."""
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully")
        return self._embeddings

    @property
    def vector_store(self) -> FAISS:
        """Lazy-load the FAISS vector store from disk, or return cached instance."""
        if self._vector_store is None:
            index_path = Config.FAISS_INDEX_DIR
            if os.path.exists(os.path.join(index_path, "index.faiss")):
                logger.info("Loading existing FAISS index from disk")
                self._vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                logger.info("No existing FAISS index found; will create on first add")
        return self._vector_store

    def _save_index(self) -> None:
        """Persist the FAISS index to disk."""
        if self._vector_store is not None:
            os.makedirs(Config.FAISS_INDEX_DIR, exist_ok=True)
            self._vector_store.save_local(Config.FAISS_INDEX_DIR)
            logger.debug("FAISS index saved to disk")

    def add_documents(self, documents: List[Document]) -> int:
        """
        Embed and store documents in the vector store.

        Args:
            documents: List of LangChain Document objects to store.

        Returns:
            Number of documents successfully added.
        """
        if not documents:
            return 0

        logger.info(f"Embedding {len(documents)} document chunks...")

        if self._vector_store is None and self.vector_store is None:
            # No existing index — create a new one from the documents
            self._vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self._vector_store.add_documents(documents)

        self._save_index()
        logger.info(f"Added {len(documents)} chunks to vector store")
        return len(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search against stored documents.

        Args:
            query: The search query string.
            k: Number of results to return.
            filter_dict: Optional metadata filters.

        Returns:
            List of most relevant Document objects.
        """
        if self.vector_store is None:
            return []

        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict

        results = self.vector_store.similarity_search(query, **kwargs)
        logger.debug(f"Search for '{query[:50]}...' returned {len(results)} results")
        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple]:
        """
        Similarity search returning documents with relevance scores.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        if self.vector_store is None:
            return []

        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=k
        )
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current document collection.

        Returns:
            Dictionary with collection metadata and counts.
        """
        try:
            if self.vector_store is None:
                return {"total_chunks": 0, "unique_sources": 0, "source_names": []}

            docstore = self._vector_store.docstore
            index_to_id = self._vector_store.index_to_docstore_id
            count = len(index_to_id)

            sources = set()
            for doc_id in index_to_id.values():
                doc = docstore.search(doc_id)
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.add(doc.metadata["source"])

            return {
                "total_chunks": count,
                "unique_sources": len(sources),
                "source_names": sorted(sources),
            }
        except Exception:
            return {"total_chunks": 0, "unique_sources": 0, "source_names": []}

    def clear_collection(self) -> None:
        """Delete all documents from the vector store."""
        try:
            index_path = Config.FAISS_INDEX_DIR
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            self._vector_store = None
            logger.info("Vector store cleared")
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")

    def delete_source(self, source_name: str) -> int:
        """
        Remove all chunks from a specific source document.

        Args:
            source_name: The source filename to remove.

        Returns:
            Number of chunks removed.
        """
        try:
            if self.vector_store is None:
                return 0

            docstore = self._vector_store.docstore
            index_to_id = self._vector_store.index_to_docstore_id

            # Find all document IDs matching this source
            ids_to_delete = []
            for doc_id in index_to_id.values():
                doc = docstore.search(doc_id)
                if (
                    hasattr(doc, "metadata")
                    and doc.metadata.get("source") == source_name
                ):
                    ids_to_delete.append(doc_id)

            if ids_to_delete:
                self._vector_store.delete(ids_to_delete)
                self._save_index()
                logger.info(f"Deleted {len(ids_to_delete)} chunks from '{source_name}'")

            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Failed to delete source '{source_name}': {e}")
            return 0
