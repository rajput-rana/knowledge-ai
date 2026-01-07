# Technical Deep Dive: Knowledge AI System

This document provides an exhaustive technical explanation of the Knowledge AI system, covering every component, data flow, design decision, and implementation detail.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Component-by-Component Analysis](#component-by-component-analysis)
4. [Data Flow and Execution Paths](#data-flow-and-execution-paths)
5. [AI Concepts Implementation](#ai-concepts-implementation)
6. [Design Patterns and Principles](#design-patterns-and-principles)
7. [Configuration and Environment](#configuration-and-environment)
8. [Error Handling and Resilience](#error-handling-and-resilience)
9. [Performance Considerations](#performance-considerations)
10. [Extension Points](#extension-points)

---

## System Overview

### What is Knowledge AI?

Knowledge AI is a production-grade Retrieval-Augmented Generation (RAG) system that combines multiple AI technologies to create an intelligent knowledge base. The system can ingest documents, create semantic representations, store them efficiently, and answer questions using retrieved context.

### Core Capabilities

1. **Document Processing**: Ingests text documents, chunks them intelligently, and stores them with metadata
2. **Semantic Search**: Uses vector embeddings to find relevant documents based on meaning, not keywords
3. **Answer Generation**: Combines retrieved context with LLM capabilities to generate accurate answers
4. **Advanced AI**: Supports agentic AI, reasoning models, and standardized tool interfaces

### Technology Stack Deep Dive

#### FastAPI Framework
- **Why FastAPI?**: Chosen for its async-first design, automatic API documentation, and type safety
- **Dependency Injection**: Uses FastAPI's dependency system to manage component lifecycle
- **Request/Response Models**: Pydantic models ensure type safety and automatic validation
- **Async Support**: All I/O operations are async, enabling high concurrency

#### FAISS Vector Store
- **What is FAISS?**: Facebook AI Similarity Search - a library for efficient similarity search
- **Why FAISS?**: Optimized for large-scale vector search with sub-millisecond query times
- **Index Type**: Using `IndexFlatL2` (Euclidean distance) - simple but effective
- **Scalability**: Can handle millions of vectors in memory, with options for disk persistence

#### OpenAI-Compatible APIs
- **Embeddings API**: Converts text to dense vectors (1536 dimensions for text-embedding-3-small)
- **Chat API**: Generates natural language responses using GPT models
- **Compatibility**: Designed to work with any OpenAI-compatible API (OpenAI, Azure OpenAI, local models)

---

## Architecture Deep Dive

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Web UI     │  │  REST API    │  │  MCP Tools   │         │
│  │  (Static)    │  │  (FastAPI)   │  │  (Protocol)  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Application Layer                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              FastAPI Dependency Injection                 │  │
│  │  • Settings (from .env)                                  │  │
│  │  • Vector Store (singleton)                              │  │
│  │  • Embedder (per request)                                │  │
│  │  • Retriever (per request)                               │  │
│  │  • RAG Pipeline (per request)                            │  │
│  │  • Agent (per request)                                    │  │
│  │  • Reasoning Router (per request)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Routes     │  │   Schemas    │  │   Handlers    │         │
│  │  /ingest     │  │  Validation  │  │  Business     │         │
│  │  /query      │  │  Type Safety │  │  Logic       │         │
│  │  /mcp/tools  │  │              │  │              │         │
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘         │
└─────────┼────────────────────────────────────┼─────────────────┘
          │                                    │
          │                                    │
┌─────────▼────────────────────────────────────▼─────────────────┐
│                    Core Processing Layer                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    RAG Pipeline                          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │ │
│  │  │ Retrieve │→ │  Context │→ │ Generate │             │ │
│  │  │  Chunks  │  │  Builder │  │  Answer  │             │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘             │ │
│  └───────┼─────────────┼──────────────┼─────────────────────┘ │
│          │             │              │                       │
│  ┌───────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐               │
│  │  Retriever  │ │  Prompt   │ │   LLM     │               │
│  │             │ │ Templates │ │  Chat     │               │
│  └──────┬──────┘ └───────────┘ └─────┬─────┘               │
│         │                            │                       │
│  ┌──────▼────────────────────────────▼──────┐               │
│  │         Agentic AI Layer                  │               │
│  │  ┌──────────┐  ┌──────────┐             │               │
│  │  │  Agent   │  │  Tools   │             │               │
│  │  │  Loop    │  │ Registry │             │               │
│  │  └────┬─────┘  └────┬─────┘             │               │
│  └───────┼─────────────┼─────────────────────┘               │
│          │             │                                     │
└──────────┼─────────────┼─────────────────────────────────────┘
           │             │
┌──────────▼─────────────▼─────────────────────────────────────┐
│              Infrastructure Layer                             │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Embedder   │  │ Vector Store │  │  Reasoning   │       │
│  │  (OpenAI)    │  │   (FAISS)    │  │   Models     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘       │
│         │                  │                                 │
│  ┌──────▼──────────────────▼──────┐                         │
│  │      External APIs             │                         │
│  │  • OpenAI Embeddings           │                         │
│  │  • OpenAI Chat                 │                         │
│  │  • Reasoning Models (o1)      │                         │
│  └────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

### Dependency Injection Architecture

FastAPI's dependency injection system is central to the architecture. Here's how it works:

#### Dependency Graph

```
get_settings()
    ↓
get_vector_store(settings) ──→ VectorStoreManager (singleton)
    ↓
get_embedder(settings) ──→ OpenAIEmbedder (per request)
    ↓
get_chat_completion(settings) ──→ ChatCompletion (per request)
    ↓
get_retriever(embedder, vector_store, settings) ──→ Retriever
    ↓
get_rag_pipeline(retriever, chat_completion, settings) ──→ RAGPipeline
    ↓
get_rag_agent(retriever, chat_completion, settings, loader) ──→ RAGAgent
    ↓
get_reasoning_model(settings) ──→ ReasoningModel
    ↓
get_reasoning_router(reasoning_model, chat_completion, settings) ──→ ReasoningRouter
```

#### Singleton Pattern for Vector Store

The vector store uses a singleton pattern to persist data across requests:

```python
class VectorStoreManager:
    _instance: Optional[FAISSVectorStore] = None
    
    @classmethod
    def get_instance(cls, settings: Settings) -> FAISSVectorStore:
        if cls._instance is None:
            cls._instance = FAISSVectorStore(...)
        return cls._instance
```

**Why Singleton?**
- Vector store must persist across requests
- FAISS index is expensive to recreate
- Documents added in one request must be queryable in the next

**Trade-offs:**
- ✅ Data persistence across requests
- ✅ Efficient memory usage
- ⚠️ Single instance (not suitable for horizontal scaling without changes)

---

## Component-by-Component Analysis

### 1. Document Ingestion Pipeline

#### Document Loader (`app/ingestion/loader.py`)

**Purpose**: Converts raw text into structured Document objects

**Key Components**:

```python
class Document:
    id: str                    # UUID or custom ID
    text: str                  # Document content
    metadata: Dict[str, Any]   # Flexible metadata storage
```

**Design Decisions**:
- **UUID Generation**: Auto-generates IDs if not provided, ensuring uniqueness
- **Metadata Flexibility**: Uses Dict[str, Any] to support any metadata structure
- **Timestamp Tracking**: Automatically adds `created_at` timestamp

**Data Flow**:
```
Raw Text → DocumentLoader.load_text() → Document Object
                                    ↓
                            Metadata Enrichment
                                    ↓
                            UUID Generation
                                    ↓
                            Timestamp Addition
```

#### Text Chunker (`app/ingestion/chunker.py`)

**Purpose**: Splits documents into smaller, overlapping chunks

**Algorithm**:
1. **Size Check**: If document < chunk_size, return as single chunk
2. **Sliding Window**: Move through text with overlap
3. **Word Boundary**: Try to break at word boundaries (not mid-word)
4. **Metadata Preservation**: Each chunk retains parent document metadata

**Chunking Strategy**:
```
Original Document (3000 chars)
├── Chunk 0: [0-1000] chars
├── Chunk 1: [800-1800] chars  (200 char overlap)
├── Chunk 2: [1600-2600] chars (200 char overlap)
└── Chunk 3: [2400-3000] chars (200 char overlap)
```

**Why Overlap?**
- Prevents information loss at chunk boundaries
- Ensures context continuity
- Improves retrieval accuracy for queries spanning boundaries

**Chunk Metadata**:
```python
{
    "chunk_index": 0,
    "total_chunks": 4,
    "parent_doc_id": "doc-123",
    "char_start": 0,
    "char_end": 1000,
    ...original_metadata...
}
```

### 2. Embedding Generation (`app/embeddings/embedder.py`)

#### OpenAI Embedder

**Purpose**: Converts text into dense vector representations

**API Interaction**:
```python
POST https://api.openai.com/v1/embeddings
{
    "model": "text-embedding-3-small",
    "input": ["text1", "text2", ...]  # Batch support
}
```

**Response Processing**:
```python
{
    "data": [
        {"embedding": [0.123, -0.456, ...]},  # 1536 dimensions
        {"embedding": [0.789, 0.012, ...]}
    ]
}
```

**Batch Processing**:
- **Why Batch?**: Reduces API calls, improves throughput
- **Batch Size**: All chunks from a document in one call
- **Error Handling**: Validates dimension matches expected (1536)

**Vector Properties**:
- **Dimension**: 1536 (for text-embedding-3-small)
- **Type**: float32 (required by FAISS)
- **Normalization**: Not normalized (FAISS handles distance calculation)

### 3. Vector Store (`app/vector_store/faiss_store.py`)

#### FAISS Implementation

**Index Type**: `IndexFlatL2`
- **L2 Distance**: Euclidean distance between vectors
- **Flat Index**: Exhaustive search (no approximation)
- **Trade-off**: Slower for very large datasets, but exact results

**Data Structure**:
```python
class FAISSVectorStore:
    index: faiss.IndexFlatL2      # FAISS index
    documents: List[Document]      # Parallel list of documents
    dimension: int                 # Vector dimension (1536)
```

**Why Parallel Lists?**
- FAISS only stores vectors, not metadata
- Documents list maintains 1:1 mapping with index positions
- Enables retrieving both vector similarity AND document content

**Add Operation**:
```python
def add(vectors: np.ndarray, documents: List[Document]):
    # vectors shape: (n_chunks, 1536)
    # documents: List of Document objects
    
    # 1. Validate dimensions
    assert vectors.shape[1] == 1536
    
    # 2. Convert to float32 (FAISS requirement)
    vectors = vectors.astype(np.float32)
    
    # 3. Add to FAISS index
    self.index.add(vectors)
    
    # 4. Store documents in parallel list
    self.documents.extend(documents)
```

**Search Operation**:
```python
def search(query_vector: np.ndarray, k: int):
    # query_vector shape: (1536,)
    
    # 1. Reshape for FAISS: (1, 1536)
    query_vector = query_vector.reshape(1, -1)
    
    # 2. FAISS search returns (distances, indices)
    distances, indices = self.index.search(query_vector, k)
    
    # 3. Convert L2 distance to similarity score
    # Lower distance = higher similarity
    similarity = 1.0 / (1.0 + distance)
    
    # 4. Map indices to documents
    results = [(self.documents[idx], similarity) 
               for idx in indices[0]]
```

**Distance to Similarity Conversion**:
- L2 distance: Lower is better (more similar)
- Similarity score: Higher is better (more similar)
- Formula: `similarity = 1 / (1 + distance)`
- Range: [0, 1] where 1 is perfect match

### 4. Retrieval System (`app/retrieval/retriever.py`)

#### Retriever Component

**Purpose**: Orchestrates query embedding and vector search

**Process**:
```python
async def retrieve(query: str):
    # 1. Embed query
    query_embedding = await embedder.embed(query)
    
    # 2. Convert to numpy array
    query_vector = np.array(query_embedding, dtype=np.float32)
    
    # 3. Search vector store
    results = vector_store.search(query_vector, k=self.top_k)
    
    # 4. Return (Document, similarity_score) tuples
    return results
```

**Top-K Selection**:
- **Default**: 5 chunks (configurable)
- **Why K?**: Balance between context size and relevance
- **Ranking**: Results sorted by similarity (highest first)

### 5. RAG Pipeline (`app/rag/pipeline.py`)

#### End-to-End RAG Process

**Step-by-Step Execution**:

```python
async def query(query: str):
    # STEP 1: Retrieve relevant chunks
    retrieved_chunks = await self.retriever.retrieve(query)
    # Returns: List[(Document, similarity_score)]
    
    # STEP 2: Build context string
    context = self._build_context(retrieved_chunks)
    # Combines chunks with separators, respects max_length
    
    # STEP 3: Build prompt
    messages = self._build_prompt(query, context)
    # Uses PromptTemplate with selected style
    
    # STEP 4: Generate answer
    answer = await self.chat_completion.complete(messages)
    # Calls OpenAI API with context
    
    # STEP 5: Extract sources
    source_documents = [doc for doc, _ in retrieved_chunks]
    
    return answer, source_documents
```

#### Context Building

**Algorithm**:
```python
def _build_context(retrieved_chunks):
    context_parts = []
    current_length = 0
    
    for doc, score in retrieved_chunks:
        chunk_text = f"[Document ID: {doc.id}]\n{doc.text}\n"
        chunk_length = len(chunk_text)
        
        # Respect max context length
        if current_length + chunk_length > max_context_length:
            break
        
        context_parts.append(chunk_text)
        current_length += chunk_length
    
    return "\n---\n\n".join(context_parts)
```

**Context Format**:
```
[Document ID: doc-123_chunk_0]
This is the first chunk of text...

---

[Document ID: doc-123_chunk_1]
This is the second chunk with overlap...

---

[Document ID: doc-456_chunk_0]
This is from a different document...
```

**Why This Format?**
- Document IDs enable source citation
- Separators (`---`) help LLM distinguish chunks
- Preserves chunk boundaries for accurate attribution

### 6. Prompt Engineering (`app/prompts/templates.py`)

#### Prompt Template System

**Supported Styles**:

1. **Standard Prompt**:
```
System: You are a helpful assistant...
User: Context: [context]
       Question: [query]
       Answer:
```

2. **Chain-of-Thought**:
```
System: Think step by step: 1) Identify info needed, 2) Find in context...
User: Context: [context]
       Question: [query]
       Let's think step by step:
       Answer:
```

3. **Few-Shot Examples**:
```
System: [Same as standard]
User: Example 1: Q: ... A: ...
       Example 2: Q: ... A: ...
       Now answer:
       Context: [context]
       Question: [query]
       Answer:
```

4. **Reasoning Prompt**:
```
System: Break down complex questions, analyze systematically...
User: Context: [context]
       Question: [query]
       Think through this step by step:
       Answer:
```

**Prompt Selection Logic**:
```python
style_map = {
    "standard": PromptStyle.STANDARD,
    "chain_of_thought": PromptStyle.CHAIN_OF_THOUGHT,
    "few_shot": PromptStyle.FEW_SHOT,
    "reasoning": PromptStyle.REASONING
}
style = style_map.get(settings.prompt_style, PromptStyle.STANDARD)
```

### 7. Agentic AI System (`app/agents/`)

#### Agent Architecture

**Base Agent Framework**:

```python
class Agent:
    tools: Dict[str, Tool]      # Available tools
    max_iterations: int         # Safety limit
    
    async def run(query: str):
        state = AgentState(query)
        
        # Agent loop
        while state.iteration < max_iterations:
            # 1. Think
            thought = await self.think(query, state)
            state.add_thought(thought)
            
            # 2. Decide action
            action = await self.decide_action(query, state)
            
            # 3. Execute tool
            result = await self.use_tool(action.tool, action.args)
            state.add_observation(result)
            
            # 4. Check if done
            if self.is_complete(state):
                break
        
        return state
```

#### RAG Agent Implementation

**Tool-Based Execution**:

```python
class RAGAgent(Agent):
    async def run(query: str):
        state = AgentState(query)
        
        # Step 1: Retrieve information
        if "rag_retrieve" in self.tools:
            result = await self.use_tool("rag_retrieve", {"query": query})
            context = self._build_context_from_results(result)
        
        # Step 2: Generate answer with context
        answer = await self._generate_answer(query, context)
        state.final_answer = answer
        
        return state
```

**Agent State Tracking**:
```python
class AgentState:
    query: str
    iteration: int
    thoughts: List[str]           # Reasoning steps
    actions: List[Dict]            # Tool calls made
    observations: List[str]        # Tool results
    final_answer: Optional[str]    # Final output
```

**Why Agentic?**
- **Autonomous Decision Making**: Agent decides which tools to use
- **Multi-Step Reasoning**: Can break complex queries into steps
- **Tool Composition**: Can chain multiple tools together
- **Traceability**: Full execution trace for debugging

#### Built-in Tools

**1. RAG Retrieval Tool**:
```python
class RAGRetrievalTool(Tool):
    async def execute({"query": "..."}):
        results = await retriever.retrieve(query)
        return {
            "results": [...],
            "count": len(results)
        }
```

**2. Document Ingestion Tool**:
```python
class DocumentIngestionTool(Tool):
    async def execute({"text": "...", "metadata": {...}}):
        document = loader.load_text(text, metadata=metadata)
        return {"doc_id": document.id}
```

**3. Query Refinement Tool**:
```python
class QueryRefinementTool(Tool):
    async def execute({"query": "...", "type": "expand"}):
        # Refines query for better search results
        return {"refined_query": "..."}
```

### 8. Reasoning Models (`app/llm/reasoning.py`)

#### Reasoning Model Support

**Purpose**: Route complex queries to reasoning-optimized models

**Model Types**:
- **Standard Models**: GPT-4o-mini (fast, general purpose)
- **Reasoning Models**: o1-preview (slower, optimized for reasoning)

**Routing Logic**:
```python
async def should_use_reasoning(query: str) -> bool:
    # Check for reasoning keywords
    reasoning_keywords = ["why", "how", "explain", "analyze", ...]
    has_keywords = any(kw in query.lower() for kw in reasoning_keywords)
    
    # Check complexity
    is_complex = len(query.split()) > 10
    
    # Check settings
    force_reasoning = settings.always_use_reasoning
    
    return has_keywords or is_complex or force_reasoning
```

**Reasoning Model API**:
```python
async def reason(query: str, context: str):
    messages = [
        {"role": "system", "content": reasoning_prompt},
        {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
    ]
    
    response = await api_call(model="o1-preview", messages=messages)
    return {
        "reasoning": response,  # Includes reasoning steps
        "answer": response      # For o1, reasoning IS the answer
    }
```

**When to Use Reasoning Models**:
- Complex analytical questions
- Multi-step problem solving
- Questions requiring logical deduction
- "Why" and "How" questions

### 9. MCP Integration (`app/mcp/`)

#### Model Context Protocol

**Purpose**: Expose RAG capabilities as standardized tools

**MCP Server Implementation**:

```python
class RAGMCPServer:
    async def list_tools():
        return [
            Tool(
                name="query_knowledge_base",
                description="...",
                inputSchema={...}
            ),
            ...
        ]
    
    async def call_tool(name: str, arguments: Dict):
        if name == "query_knowledge_base":
            # Use RAG pipeline
            answer, sources = await rag_pipeline.query(...)
            return MCPTextContent(text=answer)
```

**Tool Schema**:
```json
{
    "name": "query_knowledge_base",
    "description": "Query the knowledge base using RAG",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "prompt_style": {"type": "string", "enum": [...]},
            "top_k": {"type": "integer"}
        },
        "required": ["query"]
    }
}
```

**MCP Client**:
```python
class MCPClient:
    async def connect():
        # Verify server is accessible
        response = await httpx.get(f"{base_url}/health")
        self.connected = True
    
    async def call_tool(tool_name: str, arguments: Dict):
        response = await httpx.post(
            f"{base_url}/mcp/tools/{tool_name}",
            json=arguments
        )
        return response.json()["result"]
```

**Benefits of MCP**:
- **Standardization**: Tools follow MCP schema
- **Interoperability**: Works with MCP-compatible clients
- **Discovery**: Tools can be discovered automatically
- **Composition**: Tools can be chained together

---

## Data Flow and Execution Paths

### Path 1: Document Ingestion

```
1. HTTP Request
   POST /api/v1/ingest
   Body: {"text": "...", "metadata": {...}}
   
2. Route Handler (routes.py)
   - Validates request with Pydantic
   - Gets dependencies (vector_store, embedder)
   
3. Document Loading
   DocumentLoader.load_text()
   → Creates Document object
   → Generates UUID
   → Adds metadata
   
4. Chunking
   TextChunker.chunk_document()
   → Splits into overlapping chunks
   → Preserves metadata
   → Adds chunk-specific metadata
   
5. Embedding Generation
   OpenAIEmbedder.embed_batch()
   → HTTP POST to OpenAI API
   → Receives 1536-dim vectors
   → Converts to numpy array
   
6. Vector Storage
   FAISSVectorStore.add()
   → Adds vectors to FAISS index
   → Stores documents in parallel list
   → Optionally saves index to disk
   
7. Response
   Returns: {"doc_id": "...", "chunks_created": 3}
```

### Path 2: Standard Query

```
1. HTTP Request
   POST /api/v1/query
   Body: {"query": "What is RAG?"}
   
2. Route Handler
   - Validates request
   - Gets RAG pipeline dependency
   
3. Retrieval
   Retriever.retrieve()
   → Embeds query
   → Searches vector store
   → Returns top-k chunks
   
4. Context Building
   RAGPipeline._build_context()
   → Combines chunks
   → Respects max length
   → Formats with separators
   
5. Prompt Building
   PromptTemplate.build_prompt()
   → Selects prompt style
   → Builds system + user messages
   
6. LLM Generation
   ChatCompletion.complete()
   → HTTP POST to OpenAI API
   → Receives generated answer
   
7. Response Formatting
   → Extracts source documents
   → Formats response
   → Returns JSON
```

### Path 3: Agentic Query

```
1. HTTP Request
   POST /api/v1/query
   Body: {"query": "...", "use_agent": true}
   
2. Route Handler
   - Detects use_agent flag
   - Gets RAG agent dependency
   
3. Agent Execution
   RAGAgent.run()
   → Initial thought
   → Decides to use RAG retrieval tool
   → Calls tool
   → Observes results
   → Generates answer
   → Records trace
   
4. State Tracking
   AgentState tracks:
   - Thoughts
   - Actions (tool calls)
   - Observations (tool results)
   - Final answer
   
5. Response
   Returns answer + agent_trace
```

### Path 4: Reasoning Query

```
1. HTTP Request
   POST /api/v1/query
   Body: {"query": "...", "use_reasoning": true}
   
2. Route Handler
   - Detects use_reasoning flag
   - Gets reasoning router
   
3. Routing Decision
   ReasoningRouter.should_use_reasoning()
   → Checks keywords
   → Checks complexity
   → Returns True
   
4. Reasoning Model Call
   ReasoningModel.reason()
   → Builds reasoning prompt
   → Calls o1-preview API
   → Receives reasoning + answer
   
5. Response
   Returns answer + reasoning steps
```

---

## AI Concepts Implementation

### 1. LLM (Large Language Model)

**Implementation**: `app/llm/chat.py`

**How It Works**:
- Uses OpenAI Chat Completions API
- Supports system and user messages
- Configurable temperature, max_tokens
- Async HTTP calls for performance

**Message Format**:
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is RAG?"}
]
```

**Response Processing**:
```python
response = {
    "choices": [{
        "message": {
            "content": "RAG is..."
        }
    }]
}
answer = response["choices"][0]["message"]["content"]
```

### 2. RAG (Retrieval-Augmented Generation)

**Implementation**: `app/rag/pipeline.py`

**RAG Process**:
1. **Retrieval**: Find relevant documents
2. **Augmentation**: Add context to prompt
3. **Generation**: Generate answer with context

**Why RAG?**
- Reduces hallucination
- Enables source citation
- Allows knowledge updates without retraining
- Improves accuracy on domain-specific questions

### 3. Prompt Engineering

**Implementation**: `app/prompts/templates.py`

**Prompt Styles**:
- **Standard**: Direct question-answer
- **Chain-of-Thought**: Step-by-step reasoning
- **Few-Shot**: Examples before question
- **Reasoning**: Explicit reasoning instructions

**Impact on Output**:
- Chain-of-thought: More detailed explanations
- Few-shot: Better format consistency
- Reasoning: Structured logical flow

### 4. Vector Embeddings

**Implementation**: `app/embeddings/embedder.py`

**What Are Embeddings?**
- Dense numerical representations of text
- Capture semantic meaning
- Similar texts → similar vectors

**Embedding Properties**:
- **Dimension**: 1536 (text-embedding-3-small)
- **Type**: float32
- **Normalization**: Not normalized (distance-based)

**Semantic Search**:
```
Query: "machine learning"
Document 1: "AI and ML techniques" → High similarity
Document 2: "cooking recipes" → Low similarity
```

### 5. Agentic AI

**Implementation**: `app/agents/rag_agent.py`

**Agent Characteristics**:
- **Autonomous**: Makes own decisions
- **Tool-Using**: Can call external tools
- **Stateful**: Maintains execution state
- **Traceable**: Records all actions

**Agent Loop**:
```
Think → Decide → Act → Observe → Repeat
```

**Tool Use Pattern**:
```python
# Agent decides to use tool
action = {"tool": "rag_retrieve", "args": {"query": "..."}}

# Execute tool
result = await agent.use_tool(action["tool"], action["args"])

# Use result
context = build_context(result)
answer = generate_answer(context)
```

### 6. Reasoning Models

**Implementation**: `app/llm/reasoning.py`

**Reasoning Models**:
- **o1-preview**: OpenAI's reasoning model
- **Claude Reasoning**: Anthropic's reasoning models

**When to Use**:
- Complex analytical questions
- Multi-step problems
- Logical deduction required
- "Why" questions

**Reasoning Process**:
```
Question → Break Down → Analyze → Synthesize → Answer
```

### 7. MCP (Model Context Protocol)

**Implementation**: `app/mcp/server.py`, `app/mcp/client.py`

**MCP Purpose**:
- Standardize tool interfaces
- Enable tool discovery
- Support tool composition
- Interoperability

**MCP Flow**:
```
Client → List Tools → Select Tool → Call Tool → Get Result
```

---

## Design Patterns and Principles

### 1. Dependency Injection

**Pattern**: All dependencies injected via FastAPI

**Benefits**:
- Testability (easy to mock)
- Flexibility (swap implementations)
- No global state
- Clear dependencies

**Example**:
```python
def get_rag_pipeline(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    chat_completion: Annotated[ChatCompletion, Depends(get_chat_completion)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> RAGPipeline:
    return RAGPipeline(retriever, chat_completion, settings)
```

### 2. Protocol-Based Abstractions

**Pattern**: Define protocols for key interfaces

**Example**:
```python
class EmbedderProtocol(Protocol):
    async def embed(self, text: str) -> List[float]: ...
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
```

**Benefits**:
- Type safety
- Easy to swap implementations
- Clear contracts

### 3. Singleton Pattern

**Pattern**: Vector store uses singleton

**Implementation**:
```python
class VectorStoreManager:
    _instance: Optional[FAISSVectorStore] = None
    
    @classmethod
    def get_instance(cls, settings: Settings):
        if cls._instance is None:
            cls._instance = FAISSVectorStore(...)
        return cls._instance
```

**Why?**
- Data must persist across requests
- Expensive to recreate
- Single source of truth

### 4. Strategy Pattern

**Pattern**: Different prompt styles as strategies

**Implementation**:
```python
style_map = {
    "standard": PromptStyle.STANDARD,
    "chain_of_thought": PromptStyle.CHAIN_OF_THOUGHT,
    ...
}
style = style_map.get(settings.prompt_style)
prompt = PromptTemplate.build_prompt(..., style=style)
```

### 5. Template Method Pattern

**Pattern**: RAG pipeline defines algorithm, subclasses customize

**Flow**:
```
RAGPipeline.query()
  → retrieve()      # Template method
  → build_context() # Template method
  → build_prompt()  # Can be customized
  → generate()      # Template method
```

---

## Configuration and Environment

### Settings Management (`app/core/config.py`)

**Pydantic Settings**:
- Loads from `.env` file
- Type validation
- Default values
- Environment variable override

**Configuration Categories**:

1. **API Configuration**:
   - Base URLs
   - API keys
   - Model names

2. **Vector Store**:
   - Dimension
   - Index path
   - Persistence options

3. **Retrieval**:
   - Top-k value
   - Similarity threshold

4. **Chunking**:
   - Chunk size
   - Overlap size

5. **RAG**:
   - Max context length
   - Prompt style

6. **Agent**:
   - Max iterations
   - Enable/disable

7. **Reasoning**:
   - Model selection
   - Auto-routing

### Environment Variables

**Required**:
- `EMBEDDING_API_KEY`
- `CHAT_API_KEY`

**Optional** (with defaults):
- All other settings have sensible defaults

**Loading Order**:
1. Default values (in code)
2. `.env` file
3. Environment variables (override)

---

## Error Handling and Resilience

### Error Handling Strategy

**Levels of Error Handling**:

1. **Validation Errors** (400):
   - Invalid input
   - Missing required fields
   - Type mismatches

2. **Not Found Errors** (404):
   - Endpoint not found
   - Resource not found

3. **Server Errors** (500):
   - API failures
   - Unexpected exceptions

**Error Handling Pattern**:
```python
try:
    # Operation
    result = await operation()
    return result
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Internal error")
```

### Resilience Features

1. **API Timeouts**:
   - Embedding: 30 seconds
   - Chat: 60 seconds
   - Reasoning: 120 seconds

2. **Graceful Degradation**:
   - If reasoning model unavailable → fallback to standard
   - If agent fails → fallback to standard RAG

3. **Validation**:
   - Input validation at API boundary
   - Type checking throughout
   - Schema validation

---

## Performance Considerations

### Optimization Strategies

1. **Async Operations**:
   - All I/O is async
   - Concurrent request handling
   - Non-blocking operations

2. **Batch Processing**:
   - Embed multiple chunks in one API call
   - Reduces API overhead

3. **Vector Store Efficiency**:
   - FAISS optimized for speed
   - In-memory for fast access
   - Optional persistence

4. **Context Length Management**:
   - Limits context to prevent token waste
   - Early stopping when limit reached

### Scalability Considerations

**Current Limitations**:
- In-memory vector store
- Single instance
- No distributed architecture

**Scaling Paths**:
- **Horizontal**: Multiple instances + shared vector store
- **Vertical**: Larger memory, faster CPU
- **Distributed**: pgvector, Milvus cluster

---

## Extension Points

### Adding New Embedding Providers

1. Implement `EmbedderProtocol`
2. Create new embedder class
3. Update dependency injection
4. Configure in settings

### Adding New Vector Stores

1. Implement `VectorStoreProtocol`
2. Create new store class
3. Update dependency injection
4. Configure in settings

### Adding New Tools

1. Extend `Tool` base class
2. Implement `execute()` method
3. Register in agent
4. Add to MCP server

### Adding New Prompt Styles

1. Add to `PromptStyle` enum
2. Implement in `PromptTemplate`
3. Update settings
4. Add UI option

---

## Conclusion

This system represents a comprehensive implementation of modern AI technologies, combining RAG, agentic AI, reasoning models, and standardized protocols into a production-ready platform. The architecture is designed for extensibility, maintainability, and scalability.

Each component is carefully designed with clear responsibilities, type safety, and error handling. The system can be extended and customized without major refactoring, making it suitable for various use cases and requirements.

