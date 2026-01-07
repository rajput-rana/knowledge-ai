"""Embedding generation using OpenAI-compatible API."""
from typing import List, Protocol
import httpx
from app.core.config import Settings


class EmbedderProtocol(Protocol):
    """Protocol for embedder implementations."""
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...


class OpenAIEmbedder:
    """OpenAI-compatible embedder implementation."""
    
    def __init__(self, settings: Settings):
        """
        Initialize the embedder.
        
        Args:
            settings: Application settings
        """
        self.api_base = settings.embedding_api_base
        self.api_key = settings.embedding_api_key
        self.model = settings.embedding_model
        self.dimension = settings.vector_dimension
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            httpx.HTTPError: If API request fails
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            httpx.HTTPError: If API request fails
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        url = f"{self.api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in data["data"]]
            
            # Validate dimension
            if embeddings and len(embeddings[0]) != self.dimension:
                raise ValueError(
                    f"Expected embedding dimension {self.dimension}, "
                    f"got {len(embeddings[0])}"
                )
            
            return embeddings

