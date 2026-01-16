from langchain_core.documents import Document
from typing import List

from .parser import DoclingParser
from .chunker import MarkdownChunker
from .embedder import Embedder
from .vector_db import QdrantDB


class DocumentProcessingPipeline:
    def __init__(self, parser: DoclingParser, chunker: MarkdownChunker, embedder: Embedder, qdrant_db: QdrantDB) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.qdrant_db = qdrant_db

    def process(self, file_data: bytes, file_name: str) -> None:
        """
        Process a file from bytes.
        
        Args:
            file_data: File content as bytes
            file_name: Name of the file
        """
        pages = self.parser.parse(file_data, file_name)
        chunks = []
        for page in pages:
            chunks.extend(self.chunker.chunk(page))
        print(f"Chunked {file_name} into {len(chunks)} chunks")
        self.qdrant_db.upload(chunks)
        print(f"Uploaded {file_name} to QdrantDB")

        