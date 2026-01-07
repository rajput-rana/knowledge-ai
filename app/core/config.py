"""Configuration management using Pydantic settings."""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_title: str = Field(default="Knowledge AI API", description="API title")
    api_version: str = Field(default="0.1.0", description="API version")
    
    # OpenAI-compatible API Configuration
    embedding_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for embeddings API"
    )
    embedding_api_key: str = Field(..., description="API key for embeddings")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name"
    )
    
    chat_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for chat API"
    )
    chat_api_key: str = Field(..., description="API key for chat")
    chat_model: str = Field(
        default="gpt-4o-mini",
        description="Chat model name"
    )
    
    # Vector Store Configuration
    vector_dimension: int = Field(
        default=1536,
        description="Dimension of embedding vectors"
    )
    faiss_index_path: Optional[str] = Field(
        default=None,
        description="Optional path to persist FAISS index"
    )
    
    # Retrieval Configuration
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(
        default=1000,
        ge=100,
        description="Size of text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks in characters"
    )
    
    # RAG Configuration
    max_context_length: int = Field(
        default=4000,
        description="Maximum context length for RAG prompt"
    )
    
    # Prompt Configuration
    prompt_style: str = Field(
        default="standard",
        description="Prompt style: standard, chain_of_thought, few_shot, reasoning"
    )
    include_few_shot_examples: bool = Field(
        default=False,
        description="Include few-shot examples in prompts"
    )
    
    # Reasoning Model Configuration
    reasoning_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for reasoning API"
    )
    reasoning_api_key: str = Field(
        default="",
        description="API key for reasoning model (optional)"
    )
    reasoning_model: str = Field(
        default="o1-preview",
        description="Reasoning model name"
    )
    always_use_reasoning: bool = Field(
        default=False,
        description="Always use reasoning model (if available)"
    )
    
    # Agent Configuration
    agent_max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum iterations for agent execution"
    )
    enable_agentic_mode: bool = Field(
        default=True,
        description="Enable agentic AI mode"
    )
    
    # MCP Configuration
    enable_mcp_server: bool = Field(
        default=True,
        description="Enable MCP server to expose RAG as tools"
    )
    mcp_server_port: int = Field(
        default=8001,
        description="Port for MCP server (if using HTTP transport)"
    )
    mcp_external_servers: Optional[str] = Field(
        default=None,
        description="Comma-separated list of external MCP server configurations (JSON)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()

