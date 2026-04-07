"""
Document processing pipeline.
Handles loading, parsing, and chunking documents from multiple formats
into LangChain Document objects ready for embedding.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)

from utils.config import Config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes uploaded documents into chunked LangChain Documents.

    Supports PDF, DOCX, TXT, Markdown, and CSV files. Each document
    is split into overlapping chunks optimized for retrieval.
    """

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
    }

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_file(self, file_path: str, source_name: Optional[str] = None) -> List[Document]:
        """
        Load and chunk a single file into Document objects.

        Args:
            file_path: Path to the file on disk.
            source_name: Display name for the source (defaults to filename).

        Returns:
            List of chunked Document objects with metadata.

        Raises:
            ValueError: If the file type is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.LOADER_MAP:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(Config.SUPPORTED_EXTENSIONS)}"
            )

        loader_class = self.LOADER_MAP[ext]
        source = source_name or path.name

        logger.info(f"Loading document: {source} ({ext})")

        try:
            loader = loader_class(str(path))
            raw_documents = loader.load()
        except Exception as e:
            logger.error(f"Failed to load {source}: {e}")
            raise RuntimeError(f"Could not parse {source}: {e}")

        # Enrich metadata
        for doc in raw_documents:
            doc.metadata.update({
                "source": source,
                "file_type": ext,
                "file_size": path.stat().st_size,
            })

        # Split into chunks
        chunks = self.text_splitter.split_documents(raw_documents)

        logger.info(f"Processed {source}: {len(raw_documents)} pages -> {len(chunks)} chunks")
        return chunks

    def process_multiple(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple files and return combined chunks.

        Args:
            file_paths: List of file paths to process.

        Returns:
            Combined list of Document chunks from all files.
        """
        all_chunks = []
        errors = []

        for fp in file_paths:
            try:
                chunks = self.process_file(fp)
                all_chunks.extend(chunks)
            except Exception as e:
                errors.append((fp, str(e)))
                logger.warning(f"Skipping {fp}: {e}")

        if errors:
            logger.warning(f"Failed to process {len(errors)} file(s)")

        return all_chunks
