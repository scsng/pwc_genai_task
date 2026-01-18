"""RAG retrieval tool for document search."""

import json
from typing import List
from langchain_core.tools import tool
from langchain_core.documents import Document

from utils.rag.vector_db import QdrantDB


@tool
def search_legal_documents(vector_db : QdrantDB ,query: str) -> str:
    """
    Search through legal documents to find relevant information.
    
    This tool searches a vector database of legal documents to find information
    relevant to the user's query. Use this when you need to find specific legal
    information, case details, regulations, or any content from the document corpus.
    
    Args:
        query: The search query describing what legal information you're looking for.
               Be specific and include relevant keywords.
    
    Returns:
        A formatted string containing the most relevant document chunks found.
        Each chunk includes the content and metadata.
    
    Examples:
        search_legal_documents("What are the requirements for filing a patent?")
        search_legal_documents("statute of limitations for contract disputes")
    """
    result = vector_db.search(query)
    return format_documents(result)


def format_documents(documents: List[Document]) -> str:
    """Format a list of documents into a readable string.
    
    Args:
        documents: List of Document objects from vector search.
        
    Returns:
        Formatted string with document content and metadata.
    """
    if not documents:
        return "No relevant documents found."
    
    formatted_parts = []
    for i, doc in enumerate(documents, 1):
        formatted_chunk : dict = doc.metadata.copy()
        formatted_chunk["content"] = doc.page_content
        formatted_parts.append(json.dumps(formatted_chunk))
    return "\n".join(formatted_parts)
