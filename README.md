# Knowledge AI

A production-grade **Retrieval-Augmented Generation (RAG)** system with advanced AI capabilities including agentic AI, reasoning models, and Model Context Protocol (MCP) support.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 Overview

Knowledge AI is a comprehensive RAG system that combines multiple AI concepts:

- ✅ **LLM** - Large Language Models for generation
- ✅ **RAG** - Retrieval-Augmented Generation
- ✅ **Prompt Engineering** - Enhanced prompts with templates
- ✅ **Vector Search** - Semantic similarity search
- ✅ **Agentic AI** - Autonomous agents with tool use
- ✅ **Reasoning Models** - Large reasoning models (o1, etc.)
- ✅ **MCP** - Model Context Protocol integration

---

## ✨ Key Features

### Core RAG Capabilities
- 📚 **Document Ingestion**: Chunk documents with overlap and metadata
- 🔍 **Semantic Search**: FAISS-based vector similarity search
- 💬 **Context-Aware Answers**: Generate answers using retrieved context
- 📊 **Source Attribution**: Every answer includes document references

### Advanced AI Features
- 🤖 **Agentic AI**: Autonomous agents that use tools and make decisions
- 🧠 **Reasoning Models**: Support for reasoning-optimized models (o1, Claude reasoning)
- 🎨 **Enhanced Prompts**: Multiple prompt styles (standard, chain-of-thought, few-shot)
- 🔌 **MCP Integration**: Expose capabilities as standardized MCP tools

### Production-Ready
- 🚀 **Modern Web UI**: Clean, intuitive interface
- 🔌 **RESTful API**: Full API access with OpenAPI docs
- 🔄 **Async-First**: Built for performance with async/await
- 🧩 **Modular Design**: Easy to extend and customize
- 🔒 **Type-Safe**: Full type hints throughout
- 📝 **Well-Documented**: Comprehensive documentation

---

## 🛠 Technology Stack

### Core Framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI server
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation and settings

### AI & ML
- **[OpenAI API](https://platform.openai.com/)** - Embeddings and chat completion
- **[FAISS](https://github.com/facebookresearch/faiss)** - Vector similarity search
- **[NumPy](https://numpy.org/)** - Numerical computing

### Protocols & Standards
- **MCP (Model Context Protocol)** - Standardized tool interface
- **OpenAPI** - API documentation standard

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (or compatible API)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd knowledge-ai
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Start the server**
   ```bash
   python -m app.main
   # or
   uvicorn app.main:app --reload
   ```

6. **Access the application**
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

---

## 📖 Usage Guide

### Web UI

The easiest way to interact with Knowledge AI:

1. **Ingest Documents**
   - Go to "Ingest Document" tab
   - Paste your document text
   - Optionally add metadata as JSON
   - Click "Ingest Document"

2. **Query Knowledge Base**
   - Go to "Query Knowledge Base" tab
   - Enter your question
   - Choose options:
     - **Prompt Style**: Standard, Chain-of-Thought, Few-Shot, or Reasoning
     - **Agentic AI Mode**: Enable autonomous agent
     - **Reasoning Model**: Use reasoning-optimized model
   - Click "Get Answer"

### API Endpoints

#### Ingest Document
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text here...",
    "metadata": {"source": "docs"}
  }'
```

#### Query with Standard RAG
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "prompt_style": "chain_of_thought"
  }'
```

#### Query with Agentic AI
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "use_agent": true
  }'
```

#### Query with Reasoning Model
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain how RAG works step by step",
    "use_reasoning": true
  }'
```

### MCP Tools

#### List Available Tools
```bash
curl "http://localhost:8000/api/v1/mcp/tools"
```

#### Call MCP Tool
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/tools/query_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "prompt_style": "chain_of_thought",
    "top_k": 5
  }'
```

---

## 🏗 Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   Web UI     │              │  REST API    │            │
│  │  (Static)    │              │  Endpoints   │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼────────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
          ┌──────────────▼───────────────┐
          │      RAG Pipeline            │
          │  + Agentic AI                │
          │  + Reasoning Models          │
          │  + Enhanced Prompts          │
          └──────────────┬───────────────┘
                         │
          ┌──────────────┼───────────────┐
          │              │               │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
    │ Retriever │  │ Chat LLM  │  │  Ingest   │
    │  + Agent  │  │ + Reason  │  │  + MCP    │
    └─────┬─────┘  └───────────┘  └─────┬─────┘
          │                             │
    ┌─────▼───────┐            ┌────────▼────────┐
    │Vector Store │            │  Chunker        │
    │  (FAISS)    │            └────────┬────────┘
    └─────┬───────┘                     │
          │                      ┌──────▼──────┐
    ┌─────▼───────┐              │  Embedder   │
    │ Embeddings │◄─────────────┘              │
    └────────────┘                             │
```

### AI Concepts Covered

1. **LLM (Large Language Model)**
   - OpenAI-compatible chat completion
   - Configurable models (gpt-4o-mini, etc.)

2. **RAG (Retrieval-Augmented Generation)**
   - Document retrieval → Context building → Answer generation
   - Source attribution and citation

3. **Prompt Engineering**
   - Multiple prompt styles
   - Chain-of-thought reasoning
   - Few-shot examples
   - Custom instructions

4. **Vector Search**
   - FAISS-based similarity search
   - Configurable dimensions and metrics
   - Metadata filtering

5. **Agentic AI**
   - Autonomous agents with tool use
   - Multi-step reasoning
   - Execution trace tracking
   - Tool composition

6. **Reasoning Models**
   - Support for reasoning-optimized models (o1)
   - Automatic routing based on query complexity
   - Step-by-step reasoning display

7. **MCP (Model Context Protocol)**
   - Expose RAG as standardized tools
   - Connect to external MCP servers
   - Tool discovery and execution

---

## 📁 Project Structure

```
knowledge-ai/
├── app/
│   ├── api/                    # FastAPI routes and schemas
│   │   ├── routes.py          # Route handlers
│   │   └── schemas.py         # Pydantic models
│   │
│   ├── agents/                 # Agentic AI system
│   │   ├── base.py            # Base agent classes
│   │   ├── rag_agent.py       # RAG agent implementation
│   │   └── tools.py           # Built-in tools
│   │
│   ├── core/                   # Core utilities
│   │   ├── config.py          # Configuration
│   │   └── logging.py         # Logging setup
│   │
│   ├── embeddings/             # Embedding generation
│   │   └── embedder.py        # OpenAI embedder
│   │
│   ├── ingestion/              # Document processing
│   │   ├── loader.py          # Document loading
│   │   └── chunker.py         # Text chunking
│   │
│   ├── llm/                    # LLM integration
│   │   ├── chat.py            # Chat completion
│   │   └── reasoning.py       # Reasoning models
│   │
│   ├── mcp/                    # MCP integration
│   │   ├── server.py          # MCP server
│   │   └── client.py          # MCP client
│   │
│   ├── prompts/                # Prompt templates
│   │   └── templates.py       # Enhanced prompts
│   │
│   ├── rag/                    # RAG pipeline
│   │   └── pipeline.py        # End-to-end RAG
│   │
│   ├── retrieval/              # Retrieval logic
│   │   └── retriever.py       # Query retrieval
│   │
│   ├── static/                 # Web UI
│   │   ├── index.html         # Main page
│   │   ├── styles.css         # Styling
│   │   └── app.js             # Frontend logic
│   │
│   ├── vector_store/           # Vector storage
│   │   └── faiss_store.py     # FAISS implementation
│   │
│   └── main.py                 # FastAPI app entry point
│
├── docs/
│   ├── architecture.md         # Architecture documentation
│   └── mcp.md                  # MCP integration guide
│
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Configuration

### Environment Variables

**Required:**
- `EMBEDDING_API_KEY` - API key for embeddings
- `CHAT_API_KEY` - API key for chat completion

**Optional:**
- `PROMPT_STYLE` - Prompt style (standard, chain_of_thought, few_shot, reasoning)
- `ENABLE_AGENTIC_MODE` - Enable agentic AI (default: true)
- `REASONING_API_KEY` - API key for reasoning models
- `ENABLE_MCP_SERVER` - Enable MCP server (default: true)

See `.env.example` for all configuration options.

---

## 🧪 Examples

### Example 1: Basic RAG Query

```python
import httpx

async with httpx.AsyncClient() as client:
    # Ingest document
    await client.post("http://localhost:8000/api/v1/ingest", json={
        "text": "RAG combines retrieval with generation...",
        "metadata": {"source": "docs"}
    })
    
    # Query
    response = await client.post("http://localhost:8000/api/v1/query", json={
        "query": "What is RAG?"
    })
    print(response.json()["answer"])
```

### Example 2: Agentic Query

```python
response = await client.post("http://localhost:8000/api/v1/query", json={
    "query": "Find and summarize information about machine learning",
    "use_agent": True
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Agent trace: {result['agent_trace']}")
```

### Example 3: Using MCP Tools

```python
# List tools
tools = await client.get("http://localhost:8000/api/v1/mcp/tools")

# Call tool
result = await client.post(
    "http://localhost:8000/api/v1/mcp/tools/query_knowledge_base",
    json={"query": "What is AI?", "prompt_style": "chain_of_thought"}
)
print(result.json()["result"])
```

---

## 🔧 Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
black app/
flake8 app/
mypy app/
```

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

---

## 📚 Documentation

- [Architecture Documentation](docs/architecture.md) - System design and decisions
- [MCP Integration Guide](docs/mcp.md) - MCP protocol usage
- [API Documentation](http://localhost:8000/docs) - Interactive API docs

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

[Add your license here]

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Excellent web framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [OpenAI](https://openai.com/) - AI models and APIs
- The open-source community

---

**Built with ❤️ for the AI community**
