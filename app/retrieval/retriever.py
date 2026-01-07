"""Retrieval of relevant document chunks."""
from typing import List, Tuple
import numpy as np
from app.embeddings.embedder import OpenAIEmbedder
from app.vector_store.faiss_store import FAISSVectorStore
from app.ingestion.loader import Document


class Retriever:
    """Retrieves relevant document chunks based on query."""
    
    def __init__(self, embedder: OpenAIEmbedder, vector_store: FAISSVectorStore, top_k: int = 5):
        """
        Initialize the retriever.
        
        Args:
            embedder: Embedder instance for query embeddings
            vector_store: Vector store instance
            top_k: Number of results to retrieve
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
    
    async def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query string
            
        Returns:
            List of tuples (Document, similarity_score) sorted by relevance
        """
        # Generate query embedding
        query_embedding = await self.embedder.embed(query)
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        # Search vector store
        results = self.vector_store.search(query_vector, k=self.top_k)
        
        return results

