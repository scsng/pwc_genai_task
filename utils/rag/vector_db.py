
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, Qdrant, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

class QdrantDB:
    """Vector database wrapper for Qdrant with hybrid dense/sparse retrieval.
    
    Supports both dense embeddings and sparse (BM25) embeddings for hybrid
    search capabilities.
    """
    
    def __init__(
        self,
        collection_name: str,
        qdrant_host: str,
        embedding_model: str,
        sparse_embedding_model: str = "Qdrant/bm25",
        qdrant_api_key: str = None,
        top_k: int = 4,
    ) -> None:
        """Initialize Qdrant vector database connection.
        
        Args:
            collection_name: Name of the Qdrant collection.
            qdrant_host: URL of the Qdrant server.
            embedding_model: Model name for dense embeddings.
            sparse_embedding_model: Model name for sparse embeddings. Defaults to
                "Qdrant/bm25".
            qdrant_api_key: Optional API key for Qdrant authentication.
            top_k: Number of results to return in search queries. Defaults to 4.
        """
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_api_key = qdrant_api_key
        self.embedding_model = embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.top_k = top_k
        self.qdrant_client = self.create_qdrant_client(self.collection_name, self.qdrant_host, self.qdrant_api_key, self.sparse_embedding_model, self.embedding_model)

    def create_qdrant_client(self, collection_name: str, qdrant_host: str, qdrant_api_key: str, sparse_embedding_model: str, embedding_model: str):
        """Create or connect to Qdrant vector store with hybrid retrieval.
        
        If collection exists, connects to it. Otherwise, creates a new collection
        with empty documents.
        
        Args:
            collection_name: Name of the Qdrant collection.
            qdrant_host: URL of the Qdrant server.
            qdrant_api_key: Optional API key for authentication.
            sparse_embedding_model: Model name for sparse embeddings.
            embedding_model: Model name for dense embeddings.
            
        Returns:
            QdrantVectorStore instance configured for hybrid retrieval.
        """
        client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key, prefer_grpc=False, https=None)
        
        try:
            client.get_collection(collection_name)
            return QdrantVectorStore.from_existing_collection(
                collection_name=collection_name,
                url=qdrant_host,
                api_key=qdrant_api_key,
                sparse_embedding=FastEmbedSparse(model_name=sparse_embedding_model),
                embedding=HuggingFaceEmbeddings(model_name=embedding_model),
                retrieval_mode=RetrievalMode.HYBRID,
                prefer_grpc=False,
                https=False,
            )
        except (UnexpectedResponse, Exception):
            return QdrantVectorStore.from_documents(
                documents=[],
                collection_name=collection_name,
                url=qdrant_host,
                api_key=qdrant_api_key,
                sparse_embedding=FastEmbedSparse(model_name=sparse_embedding_model),
                embedding=HuggingFaceEmbeddings(model_name=embedding_model),
                retrieval_mode=RetrievalMode.HYBRID,
                prefer_grpc=False,
                https=False,
            )

    def upload(self, documents: List[Document]) -> None:
        """Upload documents to the vector database.
        
        Args:
            documents: List of Document objects to add to the collection.
        """
        self.qdrant_client.add_documents(documents)
    
    def search(self, query: str) -> List[Document]:
        """Search for similar documents using hybrid retrieval.
        
        Args:
            query: Search query string.
            
        Returns:
            List of top-k most similar Document objects.
        """
        return self.qdrant_client.similarity_search(query, top_k=self.top_k)
    
