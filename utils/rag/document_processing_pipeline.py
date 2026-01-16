from langchain_core.documents import Document
from typing import List

from .parser import DoclingParser
from .chunker import MarkdownChunker
from .vector_db import QdrantDB


class DocumentProcessingPipeline:
    """Pipeline for processing documents: parsing, chunking, and storing in vector DB.
    
    Orchestrates the document processing workflow by coordinating a parser to extract
    content, a chunker to split documents into manageable pieces, and a vector database
    to store the processed chunks for retrieval.
    """
    
    def __init__(self, parser: DoclingParser, chunker: MarkdownChunker, qdrant_db: QdrantDB) -> None:
        """Initialize the document processing pipeline.
        
        Args:
            parser: Parser instance for extracting content from documents.
            chunker: Chunker instance for splitting documents into smaller pieces.
            qdrant_db: Vector database instance for storing and retrieving chunks.
        """
        self.parser = parser
        self.chunker = chunker
        self.qdrant_db = qdrant_db

    def process(self, file_data: bytes, file_name: str) -> None:
        """Process a document file: parse, chunk, and upload to vector database.
        
        Takes raw file bytes, parses them into pages, chunks the content, and uploads
        the resulting chunks to the vector database for later retrieval.
        
        Args:
            file_data: Raw file content as bytes.
            file_name: Name of the file being processed.
        """
        pages = self.parser.parse(file_data, file_name)
        chunks = []
        for page in pages:
            chunks.extend(self.chunker.chunk(page))
        print(f"Chunked {file_name} into {len(chunks)} chunks")
        self.qdrant_db.upload(chunks)
        print(f"Uploaded {file_name} to QdrantDB")

        