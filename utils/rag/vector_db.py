
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, Qdrant, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

class QdrantDB:
    def __init__(
        self,
        collection_name: str,
        qdrant_host: str,
        embedding_model: str,
        sparse_embedding_model: str = "Qdrant/bm25",
        qdrant_api_key: str = None,
        top_k: int = 4,
    ) -> None:
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_api_key = qdrant_api_key
        self.embedding_model = embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.top_k = top_k
        self.qdrant_client = self.create_qdrant_client(self.collection_name, self.qdrant_host, self.qdrant_api_key, self.sparse_embedding_model, self.embedding_model)

    def create_qdrant_client(self, collection_name: str, qdrant_host: str, qdrant_api_key: str, sparse_embedding_model: str, embedding_model: str):
        # Check if collection exists
        client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key, prefer_grpc=False, https=None)
        
        try:
            # Try to get collection info
            client.get_collection(collection_name)
            # Collection exists, use from_existing_collection
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
            # Collection doesn't exist, create it with empty documents
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
        self.qdrant_client.add_documents(documents)
    
    def search(self, query: str) -> List[Document]:
        return self.qdrant_client.similarity_search(query, top_k=self.top_k)
    
