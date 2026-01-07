"""API request/response schemas."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request schema for document ingestion."""
    text: str = Field(..., description="Text content to ingest")
    doc_id: Optional[str] = Field(None, description="Optional document ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class IngestResponse(BaseModel):
    """Response schema for document ingestion."""
    doc_id: str = Field(..., description="Document ID")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Success message")


class QueryRequest(BaseModel):
    """Request schema for query."""
    query: str = Field(..., description="User query", min_length=1)
    prompt_style: Optional[str] = Field(
        default=None,
        description="Prompt style: standard, chain_of_thought, few_shot, reasoning"
    )
    use_agent: bool = Field(
        default=False,
        description="Use agentic AI mode"
    )
    use_reasoning: bool = Field(
        default=False,
        description="Force use of reasoning model"
    )


class SourceDocument(BaseModel):
    """Source document information."""
    doc_id: str = Field(..., description="Document ID")
    chunk_index: Optional[int] = Field(None, description="Chunk index if applicable")
    text_preview: str = Field(..., description="Preview of document text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Response schema for query."""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    num_sources: int = Field(..., description="Number of source documents")
    model_type: Optional[str] = Field(default=None, description="Model type used: standard, reasoning, agentic")
    reasoning: Optional[str] = Field(default=None, description="Reasoning steps if using reasoning model")
    agent_trace: Optional[Dict[str, Any]] = Field(default=None, description="Agent execution trace if using agentic mode")

