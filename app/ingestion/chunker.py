"""Text chunking with overlap."""
from typing import List
from app.ingestion.loader import Document


class TextChunker:
    """Chunks documents into smaller pieces with overlap."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunked Document instances with metadata
        """
        text = document.text
        chunks = []
        
        if len(text) <= self.chunk_size:
            # Document fits in one chunk
            chunk = Document(
                text=text,
                doc_id=f"{document.id}_chunk_0",
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "parent_doc_id": document.id
                }
            )
            chunks.append(chunk)
            return chunks
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundaries
            if end < len(text):
                # Look for the last space before the end
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = Document(
                    text=chunk_text,
                    doc_id=f"{document.id}_chunk_{chunk_index}",
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "parent_doc_id": document.id,
                        "char_start": start,
                        "char_end": end
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks

