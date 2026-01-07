# Architecture Documentation

## Overview

Knowledge AI is a production-grade RAG (Retrieval-Augmented Generation) system designed with scalability, maintainability, and extensibility in mind. This document describes the system architecture, data flow, key design decisions, and scaling considerations.

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                         │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ POST /ingest │              │ POST /query  │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼────────────────────┘
          │                              │
          │                              │
┌─────────▼──────────────────────────────▼────────────────────┐
│                    Application Layer                        │
│                                                              │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   Ingest     │              │ RAG Pipeline  │            │
│  │   Flow       │              │               │            │
│  └──────┬───────┘              └───────┬───────┘            │
│         │                              │                    │
│    ┌────▼────┐                    ┌────▼────┐              │
│    │ Chunker │                    │Retriever│              │
│    └────┬────┘                    └────┬────┘              │
│         │                              │                    │
└─────────┼──────────────────────────────┼────────────────────┘
          │                              │
          │                              │
┌─────────▼──────────────────────────────▼────────────────────┐
│                    Infrastructure Layer                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Embedder    │  │ Vector Store │  │ Chat LLM     │     │
│  │  (OpenAI)    │  │   (FAISS)    │  │  (OpenAI)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Ingestion Flow

1. **Request Reception**: FastAPI receives POST request to `/ingest` with text and optional metadata
2. **Document Loading**: `DocumentLoader` creates a `Document` object with ID and metadata
3. **Chunking**: `TextChunker` splits document into overlapping chunks
4. **Embedding**: `OpenAIEmbedder` generates embeddings for all chunks in batch
5. **Storage**: `FAISSVectorStore` stores vectors and associated document metadata
6. **Response**: Returns document ID and chunk count

### Query Flow

1. **Request Reception**: FastAPI receives POST request to `/query` with user question
2. **Query Embedding**: `Retriever` uses `OpenAIEmbedder` to embed the query
3. **Vector Search**: `FAISSVectorStore` performs similarity search returning top-k chunks
4. **Context Building**: `RAGPipeline` combines retrieved chunks into context string
5. **Prompt Construction**: System and user prompts are built with context
6. **LLM Generation**: `ChatCompletion` generates answer using retrieved context
7. **Response**: Returns answer with source document references

## Key Design Decisions

### 1. Dependency Injection

**Decision**: Use FastAPI's dependency injection system instead of global state.

**Rationale**: 
- Enables testability and mocking
- Allows configuration per request
- Prevents shared state issues
- Follows SOLID principles

**Implementation**: All components (vector store, embedder, retriever, etc.) are created via dependency functions.

### 2. Protocol-Based Abstractions

**Decision**: Define protocols (`EmbedderProtocol`, `VectorStoreProtocol`) for key interfaces.

**Rationale**:
- Allows swapping implementations without changing dependent code
- Enables testing with mock implementations
- Supports future migrations (e.g., FAISS → pgvector)

**Example**: `FAISSVectorStore` implements vector store interface, can be replaced with `PgVectorStore` without changing retrieval logic.

### 3. Async-First Design

**Decision**: Use async/await for I/O operations (API calls, embeddings).

**Rationale**:
- Better performance for concurrent requests
- Non-blocking I/O operations
- Scalability for production workloads

**Implementation**: Embedder and chat completion use `httpx.AsyncClient` for async HTTP requests.

### 4. Document Metadata Preservation

**Decision**: Store rich metadata with each document chunk.

**Rationale**:
- Enables source attribution in responses
- Supports filtering and advanced retrieval
- Maintains document lineage

**Implementation**: Each chunk includes parent document ID, chunk index, and custom metadata.

### 5. Configurable Chunking

**Decision**: Make chunk size and overlap configurable via environment variables.

**Rationale**:
- Different document types require different chunking strategies
- Allows optimization without code changes
- Supports experimentation

### 6. Error Handling Strategy

**Decision**: Use HTTP exceptions with appropriate status codes and error messages.

**Rationale**:
- Clear error communication to API consumers
- Proper logging for debugging
- Graceful degradation

**Implementation**: Try-except blocks in routes with specific error types mapped to HTTP status codes.

## Module Responsibilities

### `app/core/`
- **config.py**: Centralized configuration management using Pydantic settings
- **logging.py**: Logging setup and configuration

### `app/ingestion/`
- **loader.py**: Document loading and metadata assignment
- **chunker.py**: Text chunking with configurable overlap

### `app/embeddings/`
- **embedder.py**: OpenAI-compatible embedding generation with batching support

### `app/vector_store/`
- **faiss_store.py**: FAISS-based vector storage with metadata tracking

### `app/retrieval/`
- **retriever.py**: Query embedding and similarity search orchestration

### `app/llm/`
- **chat.py**: OpenAI-compatible chat completion client

### `app/rag/`
- **pipeline.py**: End-to-end RAG pipeline combining retrieval and generation

### `app/api/`
- **routes.py**: FastAPI route handlers with dependency injection
- **schemas.py**: Pydantic models for request/response validation

## Scaling Considerations

### Current Limitations

1. **In-Memory Vector Store**: FAISS index is stored in memory, limiting dataset size
2. **No Persistence**: Documents are lost on server restart (unless FAISS_INDEX_PATH is set)
3. **Single Instance**: No distributed architecture support
4. **No Caching**: Every query hits the embedding and LLM APIs

### Scaling Paths

#### 1. Vector Store Scaling

**Current**: FAISS in-memory index

**Options**:
- **pgvector**: PostgreSQL extension for production-grade vector storage
- **Milvus**: Distributed vector database with horizontal scaling
- **Pinecone/Weaviate**: Managed vector databases

**Migration Path**: Implement `VectorStoreProtocol` with new backend, swap dependency.

#### 2. Embedding Provider Scaling

**Current**: Single OpenAI-compatible API

**Options**:
- **Multiple Providers**: Support multiple embedding APIs with fallback
- **Local Models**: Use sentence-transformers for on-premise deployments
- **Caching Layer**: Cache embeddings to reduce API calls

**Migration Path**: Extend `EmbedderProtocol` implementations.

#### 3. Persistence Layer

**Current**: Optional FAISS file persistence

**Options**:
- **Database Backend**: Store documents and metadata in PostgreSQL/MongoDB
- **Object Storage**: Store FAISS indices in S3/GCS
- **Hybrid**: Metadata in DB, vectors in specialized store

**Migration Path**: Add persistence layer abstraction, implement for chosen backend.

#### 4. Multi-Tenancy

**Current**: Single-tenant system

**Options**:
- **Tenant Isolation**: Add tenant_id to all operations
- **Separate Indices**: Per-tenant vector stores
- **Row-Level Security**: Database-level tenant filtering

**Migration Path**: Add tenant context to dependency injection, filter operations by tenant.

#### 5. Performance Optimization

**Current**: Synchronous processing

**Options**:
- **Background Jobs**: Async document ingestion
- **Caching**: Redis cache for frequent queries
- **Batch Processing**: Batch multiple queries
- **Streaming**: Stream LLM responses

**Migration Path**: Add async job queue (Celery/RQ), implement caching layer.

#### 6. Monitoring & Observability

**Current**: Basic logging

**Options**:
- **Metrics**: Prometheus metrics for latency, throughput
- **Tracing**: OpenTelemetry for distributed tracing
- **Alerting**: Alerts for errors, latency spikes

**Migration Path**: Add instrumentation middleware, integrate observability tools.

## Security Considerations

### Current State
- No authentication/authorization
- API keys stored in environment variables
- No input sanitization beyond Pydantic validation

### Future Enhancements
- API key authentication
- Rate limiting
- Input validation and sanitization
- Audit logging
- Secrets management (Vault, AWS Secrets Manager)

## Testing Strategy

### Unit Tests
- Test each module in isolation
- Mock external dependencies (API calls, vector store)
- Test error handling paths

### Integration Tests
- Test full ingestion flow
- Test full query flow
- Test with real FAISS index

### Performance Tests
- Load testing for concurrent requests
- Latency benchmarks
- Throughput measurements

## Deployment Considerations

### Development
- Single instance with in-memory FAISS
- Local .env file for configuration

### Production
- Containerized deployment (Docker)
- Environment-specific configuration
- Persistent vector store
- Horizontal scaling with load balancer
- Health checks and graceful shutdown

## Conclusion

This architecture provides a solid foundation for a production RAG system while maintaining simplicity and extensibility. The modular design allows incremental scaling and feature additions without major refactoring.

