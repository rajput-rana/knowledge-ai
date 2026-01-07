"""FAISS-based vector store implementation."""
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from app.ingestion.loader import Document


class VectorStoreProtocol:
    """Protocol for vector store implementations."""
    
    def add(self, vectors: np.ndarray, documents: List[Document]) -> None:
        """Add vectors and documents to the store."""
        ...
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[Document, float]]:
        """Search for similar vectors."""
        ...


class FAISSVectorStore:
    """FAISS-based vector store with metadata support."""
    
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of embedding vectors
            index_path: Optional path to load/save FAISS index
        """
        self.dimension = dimension
        self.index_path = index_path
        
        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store documents by their index position
        self.documents: List[Document] = []
        
        # Load existing index if path provided
        if index_path:
            try:
                self.index = faiss.read_index(index_path)
                # Note: In production, you'd also need to load documents
                # This is a simplified version
            except FileNotFoundError:
                pass
    
    def add(self, vectors: np.ndarray, documents: List[Document]) -> None:
        """
        Add vectors and documents to the store.
        
        Args:
            vectors: Numpy array of shape (n, dimension)
            documents: List of Document objects corresponding to vectors
            
        Raises:
            ValueError: If dimensions don't match or lengths don't match
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match "
                f"store dimension {self.dimension}"
            )
        
        if len(vectors) != len(documents):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) does not match "
                f"number of documents ({len(documents)})"
            )
        
        # Ensure vectors are float32 for FAISS
        vectors = vectors.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Store documents
        self.documents.extend(documents)
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[Document, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector of shape (dimension,)
            k: Number of results to return
            
        Returns:
            List of tuples (Document, distance_score)
            
        Raises:
            ValueError: If store is empty or dimension doesn't match
        """
        if self.index.ntotal == 0:
            return []
        
        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[0]} does not match "
                f"store dimension {self.dimension}"
            )
        
        # Ensure query is float32 and reshape for FAISS
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Convert to results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + distance)
                results.append((self.documents[idx], similarity))
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            path: Optional path to save (uses self.index_path if not provided)
        """
        save_path = path or self.index_path
        if save_path:
            faiss.write_index(self.index, save_path)
    
    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return self.index.ntotal

