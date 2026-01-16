
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import FakeEmbeddings  # Replace with actual embeddings in real usage

class QdrantDB:
    def __init__(
        self,
        collection_name: str,
        qdrant_host: str,
        qdrant_api_key: str = None,
    ) -> None:
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_api_key = qdrant_api_key

    def upload(self, documents: List[Document]) -> None:
        Qdrant.from_documents(
            documents,
            embedding=self.embedder,
            url=self.qdrant_host,
            collection_name=self.collection_name,
            api_key=self.qdrant_api_key
        )


    def search(self, query: str, k: int = 10) -> List[Document]:
        return Qdrant.similarity_search(self.collection_name, query, k)

    
