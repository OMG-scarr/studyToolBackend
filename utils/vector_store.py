"""
Vector store management using ChromaDB.
Handles document embedding, storage, and similarity search
with persistent local storage.
"""

import logging
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from utils.config import Config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages the ChromaDB vector store for document embeddings.

    Uses HuggingFace sentence-transformers for local embedding
    generation (no API keys required). Persists data to disk
    so knowledge survives between sessions.
    """

    def __init__(self):
        self._embeddings = None
        self._vector_store = None
        self._client = None

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
    def client(self) -> chromadb.ClientAPI:
        """Lazy-load the ChromaDB persistent client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=Config.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def vector_store(self) -> Chroma:
        """Lazy-load the LangChain Chroma wrapper."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                client=self.client,
                collection_name=Config.COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
        return self._vector_store

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
        self.vector_store.add_documents(documents)
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
            collection = self.client.get_collection(Config.COLLECTION_NAME)
            count = collection.count()

            # Extract unique sources
            if count > 0:
                all_metadata = collection.get(include=["metadatas"])
                sources = set()
                for meta in all_metadata.get("metadatas", []):
                    if meta and "source" in meta:
                        sources.add(meta["source"])
            else:
                sources = set()

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
            self.client.delete_collection(Config.COLLECTION_NAME)
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
            collection = self.client.get_collection(Config.COLLECTION_NAME)
            results = collection.get(
                where={"source": source_name},
                include=["metadatas"],
            )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks from '{source_name}'")
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Failed to delete source '{source_name}': {e}")
            return 0
