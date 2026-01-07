# Sample Documents and Queries

This file contains sample documents and queries for testing the Knowledge AI system.

---

## Sample Documents

### Document 1: RAG Overview

```
Retrieval-Augmented Generation (RAG) is a technique that combines the power of information retrieval with large language models. RAG works by first retrieving relevant documents from a knowledge base, then using those documents as context for the language model to generate accurate and grounded answers.

The RAG process involves three main steps:
1. Retrieval: Given a user query, the system searches a vector database to find the most relevant document chunks using semantic similarity.
2. Augmentation: The retrieved chunks are combined into a context window that provides background information.
3. Generation: The language model generates an answer based on both the user query and the retrieved context.

RAG addresses key limitations of LLMs, including hallucination, lack of up-to-date information, and inability to cite sources. By grounding responses in retrieved documents, RAG ensures answers are factual and traceable.
```

**Metadata:**
```json
{
  "source": "docs",
  "topic": "rag",
  "type": "overview"
}
```

---

### Document 2: Vector Embeddings

```
Vector embeddings are dense numerical representations of text that capture semantic meaning. Unlike traditional keyword-based search, embeddings allow systems to find documents based on meaning rather than exact word matches.

Embeddings are created using neural networks trained on large text corpora. Popular embedding models include OpenAI's text-embedding-3-small (1536 dimensions) and text-embedding-3-large (3072 dimensions). These models convert text into high-dimensional vectors where semantically similar texts are positioned close together in the vector space.

Vector similarity search uses distance metrics like cosine similarity or Euclidean distance to find the most relevant documents. FAISS (Facebook AI Similarity Search) is a popular library for efficient vector search at scale, supporting billions of vectors with sub-millisecond query times.
```

**Metadata:**
```json
{
  "source": "docs",
  "topic": "embeddings",
  "type": "technical"
}
```

---

### Document 3: FastAPI Framework

```
FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints. It was created by Sebastián Ramírez and released in 2018. FastAPI is designed to be easy to use and learn, fast to code, ready for production, and based on open standards.

Key features of FastAPI include:
- Automatic interactive API documentation (Swagger UI and ReDoc)
- Type validation using Pydantic
- Async/await support for high performance
- Dependency injection system
- Built-in data validation and serialization
- Support for OpenAPI and JSON Schema

FastAPI is built on top of Starlette for web parts and Pydantic for data validation. It's one of the fastest Python frameworks available, comparable to Node.js and Go in terms of performance. The framework is particularly popular for building microservices and APIs that need to handle high concurrency.
```

**Metadata:**
```json
{
  "source": "docs",
  "topic": "fastapi",
  "type": "framework"
}
```

---

### Document 4: Agentic AI

```
Agentic AI refers to artificial intelligence systems that can autonomously make decisions, use tools, and take actions to achieve goals. Unlike traditional AI that responds to prompts, agentic AI can plan multi-step tasks, reason about actions, and adapt its strategy based on outcomes.

Key characteristics of agentic AI include:
- Tool use: Ability to interact with external systems and APIs
- Planning: Breaking down complex tasks into sub-tasks
- Reasoning: Thinking through problems step by step
- Memory: Maintaining context across multiple interactions
- Reflection: Evaluating and improving its own performance

Agentic AI systems typically use frameworks like LangChain, AutoGPT, or custom implementations. They combine large language models with tool-calling capabilities, allowing them to perform actions like web searches, database queries, code execution, and API calls. This makes them powerful for complex, multi-step problem solving.
```

**Metadata:**
```json
{
  "source": "docs",
  "topic": "agentic_ai",
  "type": "concept"
}
```

---

### Document 5: Model Context Protocol (MCP)

```
Model Context Protocol (MCP) is a standardized protocol introduced by Anthropic for connecting AI models to external tools, services, and data sources. MCP provides a universal interface that enables AI applications to discover and use capabilities from various providers.

MCP defines a standard way for:
- Tools to expose their capabilities with schemas
- AI applications to discover available tools
- Tools to be called with standardized parameters
- Results to be returned in a consistent format

The protocol supports multiple transport mechanisms including stdio and HTTP. MCP enables interoperability between different AI systems and tool providers, making it easier to build composable AI applications. Major platforms like Claude Desktop, ChatGPT, and Gemini have adopted MCP for tool integration.
```

**Metadata:**
```json
{
  "source": "docs",
  "topic": "mcp",
  "type": "protocol"
}
```

---

## Sample Queries

### Basic RAG Queries

#### Query 1: Simple Fact Question
```
What is RAG?
```

#### Query 2: How/Why Question
```
How does RAG work?
```

#### Query 3: Comparison Question
```
What are the differences between RAG and traditional language models?
```

#### Query 4: Step-by-Step Explanation
```
Explain the RAG process step by step
```

#### Query 5: Feature Question
```
What are the key features of FastAPI?
```

### Advanced Queries (for Agentic/Reasoning Modes)

#### Query 6: Multi-Step Reasoning
```
Why is RAG better than using LLMs alone? Explain the reasoning step by step.
```

#### Query 7: Complex Analysis
```
Compare and contrast RAG, vector embeddings, and agentic AI. How do they work together?
```

#### Query 8: Synthesis Question
```
How can I build a production RAG system using FastAPI, vector embeddings, and agentic AI?
```

#### Query 9: Technical Deep Dive
```
Explain how vector embeddings enable semantic search in RAG systems. What makes them better than keyword search?
```

#### Query 10: Architecture Question
```
What is the complete architecture of a RAG system that uses MCP for tool integration?
```

---

## Testing Workflow

### Step 1: Ingest Documents

#### Ingest Document 1 (RAG Overview)
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Retrieval-Augmented Generation (RAG) is a technique that combines the power of information retrieval with large language models. RAG works by first retrieving relevant documents from a knowledge base, then using those documents as context for the language model to generate accurate and grounded answers.\n\nThe RAG process involves three main steps:\n1. Retrieval: Given a user query, the system searches a vector database to find the most relevant document chunks using semantic similarity.\n2. Augmentation: The retrieved chunks are combined into a context window that provides background information.\n3. Generation: The language model generates an answer based on both the user query and the retrieved context.\n\nRAG addresses key limitations of LLMs, including hallucination, lack of up-to-date information, and inability to cite sources. By grounding responses in retrieved documents, RAG ensures answers are factual and traceable.",
    "metadata": {
      "source": "docs",
      "topic": "rag",
      "type": "overview"
    }
  }'
```

#### Ingest Document 2 (Vector Embeddings)
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Vector embeddings are dense numerical representations of text that capture semantic meaning. Unlike traditional keyword-based search, embeddings allow systems to find documents based on meaning rather than exact word matches.\n\nEmbeddings are created using neural networks trained on large text corpora. Popular embedding models include OpenAI'\''s text-embedding-3-small (1536 dimensions) and text-embedding-3-large (3072 dimensions). These models convert text into high-dimensional vectors where semantically similar texts are positioned close together in the vector space.\n\nVector similarity search uses distance metrics like cosine similarity or Euclidean distance to find the most relevant documents. FAISS (Facebook AI Similarity Search) is a popular library for efficient vector search at scale, supporting billions of vectors with sub-millisecond query times.",
    "metadata": {
      "source": "docs",
      "topic": "embeddings",
      "type": "technical"
    }
  }'
```

#### Ingest Document 3 (FastAPI)
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints. It was created by Sebastián Ramírez and released in 2018. FastAPI is designed to be easy to use and learn, fast to code, ready for production, and based on open standards.\n\nKey features of FastAPI include:\n- Automatic interactive API documentation (Swagger UI and ReDoc)\n- Type validation using Pydantic\n- Async/await support for high performance\n- Dependency injection system\n- Built-in data validation and serialization\n- Support for OpenAPI and JSON Schema\n\nFastAPI is built on top of Starlette for web parts and Pydantic for data validation. It'\''s one of the fastest Python frameworks available, comparable to Node.js and Go in terms of performance. The framework is particularly popular for building microservices and APIs that need to handle high concurrency.",
    "metadata": {
      "source": "docs",
      "topic": "fastapi",
      "type": "framework"
    }
  }'
```

#### Ingest Document 4 (Agentic AI)
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Agentic AI refers to artificial intelligence systems that can autonomously make decisions, use tools, and take actions to achieve goals. Unlike traditional AI that responds to prompts, agentic AI can plan multi-step tasks, reason about actions, and adapt its strategy based on outcomes.\n\nKey characteristics of agentic AI include:\n- Tool use: Ability to interact with external systems and APIs\n- Planning: Breaking down complex tasks into sub-tasks\n- Reasoning: Thinking through problems step by step\n- Memory: Maintaining context across multiple interactions\n- Reflection: Evaluating and improving its own performance\n\nAgentic AI systems typically use frameworks like LangChain, AutoGPT, or custom implementations. They combine large language models with tool-calling capabilities, allowing them to perform actions like web searches, database queries, code execution, and API calls. This makes them powerful for complex, multi-step problem solving.",
    "metadata": {
      "source": "docs",
      "topic": "agentic_ai",
      "type": "concept"
    }
  }'
```

#### Ingest Document 5 (MCP)
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Model Context Protocol (MCP) is a standardized protocol introduced by Anthropic for connecting AI models to external tools, services, and data sources. MCP provides a universal interface that enables AI applications to discover and use capabilities from various providers.\n\nMCP defines a standard way for:\n- Tools to expose their capabilities with schemas\n- AI applications to discover available tools\n- Tools to be called with standardized parameters\n- Results to be returned in a consistent format\n\nThe protocol supports multiple transport mechanisms including stdio and HTTP. MCP enables interoperability between different AI systems and tool providers, making it easier to build composable AI applications. Major platforms like Claude Desktop, ChatGPT, and Gemini have adopted MCP for tool integration.",
    "metadata": {
      "source": "docs",
      "topic": "mcp",
      "type": "protocol"
    }
  }'
```

### Step 2: Test Standard RAG Queries

#### Query 1: Simple Question
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

#### Query 2: Chain-of-Thought Prompt
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does RAG work?",
    "prompt_style": "chain_of_thought"
  }'
```

#### Query 3: Few-Shot Examples
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of FastAPI?",
    "prompt_style": "few_shot"
  }'
```

### Step 3: Test Agentic Mode

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find and summarize information about how RAG, embeddings, and agentic AI work together",
    "use_agent": true
  }'
```

### Step 4: Test Reasoning Mode

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why is RAG better than using LLMs alone? Explain step by step.",
    "use_reasoning": true
  }'
```

### Step 5: Test MCP Tools

#### Query via MCP Tool
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/tools/query_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of FastAPI?",
    "prompt_style": "chain_of_thought",
    "top_k": 3
  }'
```

#### Search Knowledge Base
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/tools/search_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vector embeddings",
    "top_k": 5
  }'
```

#### Get Knowledge Base Statistics
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/tools/get_knowledge_base_stats" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Python Testing Script

You can also use this Python script to test the system:

```python
import asyncio
import httpx

BASE_URL = "http://localhost:8000/api/v1"

# Sample documents
DOCUMENTS = [
    {
        "text": "Retrieval-Augmented Generation (RAG) is a technique that combines the power of information retrieval with large language models...",
        "metadata": {"source": "docs", "topic": "rag", "type": "overview"}
    },
    # Add other documents here
]

# Sample queries
QUERIES = [
    "What is RAG?",
    "How does RAG work?",
    "What are the key features of FastAPI?",
]

async def test_system():
    async with httpx.AsyncClient() as client:
        # Ingest documents
        print("Ingesting documents...")
        for doc in DOCUMENTS:
            response = await client.post(f"{BASE_URL}/ingest", json=doc)
            print(f"  Ingested: {response.json()['doc_id']}")
        
        # Test queries
        print("\nTesting queries...")
        for query in QUERIES:
            response = await client.post(f"{BASE_URL}/query", json={"query": query})
            result = response.json()
            print(f"\nQuery: {query}")
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Sources: {result['num_sources']}")

if __name__ == "__main__":
    asyncio.run(test_system())
```

---

## Expected Results

After ingesting all 5 documents, you should be able to:

1. ✅ Get accurate answers about RAG, embeddings, FastAPI, agentic AI, and MCP
2. ✅ See source citations for each answer
3. ✅ Test different prompt styles and see how they affect responses
4. ✅ Use agentic mode for complex multi-step queries
5. ✅ Use reasoning mode for analytical questions
6. ✅ Access all capabilities via MCP tools

---

## Quick Test Checklist

- [ ] Server is running (`curl http://localhost:8000/health`)
- [ ] Ingest all 5 sample documents
- [ ] Test basic query: "What is RAG?"
- [ ] Test chain-of-thought: "How does RAG work?" with `prompt_style: "chain_of_thought"`
- [ ] Test agentic mode: Complex query with `use_agent: true`
- [ ] Test reasoning mode: Analytical query with `use_reasoning: true`
- [ ] Test MCP tools: List tools and call `query_knowledge_base`
- [ ] Check knowledge base stats via MCP tool

---

## Notes

- Make sure your `.env` file has valid API keys before testing
- The server should be running on `http://localhost:8000`
- All queries return source documents for verification
- Different prompt styles may produce different answer formats
- Agentic mode shows execution traces in the response
- Reasoning mode includes step-by-step reasoning when available

