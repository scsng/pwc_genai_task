from docling.chunking import HybridChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
class MarkdownChunker:
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 20):
        """Initialize markdown chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between adjacent chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: Document) -> List[Document]:
        """Split document into chunks using markdown headers and size limits.

        First splits by markdown headers (h1, h2, h3) to preserve structure,
        then further splits chunks exceeding max_chunk_size. Headers are
        prepended to each chunk's content for context.

        Args:
            doc: Document to chunk.

        Returns:
            List of Document chunks with header metadata and headers in content.
        """
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
        )
        metadata = doc.metadata
        header_chunks = header_splitter.split_text(doc.page_content)
        for chunk in header_chunks:
            chunk.metadata.update(metadata)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". "]  # Splits at paragraph, line, then sentence boundaries
        )
        
        def text_enrichment(chunk: Document) -> Document:
            """Prepend header hierarchy from metadata to chunk content.

            Extracts h1, h2, h3 headers from chunk metadata and prepends them
            as markdown headers to the chunk's page_content for better context.

            Args:
                chunk: Document chunk to enrich.

            Returns:
                Document chunk with headers prepended to page_content.
            """
            header_mapping = {
                "h1": "#",
                "h2": "##",
                "h3": "###"
            }
            headers = []
            # Extract headers in hierarchical order
            for header_key in ["h1", "h2", "h3"]:
                if header_key in chunk.metadata and chunk.metadata[header_key]:
                    header_value = chunk.metadata[header_key]
                    markdown_prefix = header_mapping[header_key]
                    headers.append(f"{markdown_prefix} {header_value}")
            
            if headers:
                header_text = "\n".join(headers) + "\n\n"
                chunk.page_content = header_text + chunk.page_content
            
            return chunk
        
        final_chunks = []
        for chunk in header_chunks:
            chunk = text_enrichment(chunk)
            
            if len(chunk.page_content) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                sub_chunks = text_splitter.split_documents([chunk])
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update(chunk.metadata)
                    sub_chunk = text_enrichment(sub_chunk)
                final_chunks.extend(sub_chunks)
        
        return final_chunks
        