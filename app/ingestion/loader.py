"""Document loading and preprocessing."""
from typing import Dict, List, Optional
from datetime import datetime
import uuid


class Document:
    """Represents a document with metadata."""
    
    def __init__(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, any]] = None
    ):
        """
        Initialize a document.
        
        Args:
            text: Document text content
            doc_id: Optional document ID (generated if not provided)
            metadata: Optional metadata dictionary
        """
        self.id = doc_id or str(uuid.uuid4())
        self.text = text
        self.metadata = metadata or {}
        self.metadata["created_at"] = datetime.utcnow().isoformat()
    
    def __repr__(self) -> str:
        return f"Document(id={self.id}, text_length={len(self.text)})"


class DocumentLoader:
    """Loads and prepares documents for ingestion."""
    
    def load_text(self, text: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, any]] = None) -> Document:
        """
        Load text into a Document object.
        
        Args:
            text: Text content to load
            doc_id: Optional document ID
            metadata: Optional metadata dictionary
            
        Returns:
            Document instance
            
        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        return Document(
            text=text.strip(),
            doc_id=doc_id,
            metadata=metadata or {}
        )
    
    def load_batch(self, texts: List[str], metadata_list: Optional[List[Dict[str, any]]] = None) -> List[Document]:
        """
        Load multiple texts into Document objects.
        
        Args:
            texts: List of text contents
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of Document instances
        """
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        if len(metadata_list) != len(texts):
            raise ValueError("metadata_list length must match texts length")
        
        return [
            self.load_text(text, metadata=meta)
            for text, meta in zip(texts, metadata_list)
        ]

