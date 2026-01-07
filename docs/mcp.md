# Model Context Protocol (MCP) Integration

Knowledge AI includes MCP (Model Context Protocol) support, allowing you to expose RAG capabilities as standardized tools that can be used by AI applications and connect to external MCP servers.

## What is MCP?

Model Context Protocol (MCP) is a standardized protocol for connecting AI models to external tools, services, and data sources. It enables:

- **Standardized Tool Interface**: Expose capabilities as tools with well-defined schemas
- **Interoperability**: Connect with other MCP-compatible systems
- **Extensibility**: Easily add new tools and capabilities

## MCP Server

Knowledge AI exposes its RAG capabilities as MCP tools via a REST API that follows MCP principles.

### Available Tools

#### 1. `query_knowledge_base`

Query the knowledge base using RAG. Retrieves relevant documents and generates an answer.

**Parameters:**
- `query` (required): The question to ask
- `prompt_style` (optional): Prompt style - "standard", "chain_of_thought", "few_shot", or "reasoning"
- `top_k` (optional): Number of documents to retrieve (default: 5, max: 20)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/tools/query_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "prompt_style": "chain_of_thought",
    "top_k": 5
  }'
```

#### 2. `search_knowledge_base`

Search the knowledge base for relevant documents without generating an answer.

**Parameters:**
- `query` (required): Search query
- `top_k` (optional): Number of results to return (default: 5, max: 20)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/tools/search_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 10
  }'
```

#### 3. `ingest_document`

Add a document to the knowledge base. The document will be chunked, embedded, and stored.

**Parameters:**
- `text` (required): Document text content
- `doc_id` (optional): Document ID
- `metadata` (optional): Metadata dictionary

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/tools/ingest_document" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "RAG combines retrieval with generation...",
    "metadata": {"source": "docs", "topic": "ai"}
  }'
```

#### 4. `get_knowledge_base_stats`

Get statistics about the knowledge base.

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/tools/get_knowledge_base_stats" \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Listing Available Tools

```bash
curl "http://localhost:8000/api/v1/mcp/tools"
```

Response:
```json
{
  "tools": [
    {
      "name": "query_knowledge_base",
      "description": "Query the knowledge base using RAG...",
      "inputSchema": {...}
    },
    ...
  ]
}
```

## Configuration

Enable/disable MCP server in `.env`:

```bash
ENABLE_MCP_SERVER=true
MCP_SERVER_PORT=8001
```

## Using MCP Tools in AI Applications

MCP tools can be used by AI applications that support MCP protocol:

1. **List available tools**: `GET /api/v1/mcp/tools`
2. **Call a tool**: `POST /api/v1/mcp/tools/{tool_name}` with JSON body containing tool arguments
3. **Get result**: Response contains tool execution result

## MCP Client

Knowledge AI also includes an MCP client for connecting to external MCP servers (implementation in `app/mcp/client.py`). This allows the system to use tools from other MCP-compatible services.

## Integration Examples

### Using with Claude Desktop

You can configure Claude Desktop to use Knowledge AI's MCP server by adding it to your MCP configuration.

### Using with Custom AI Applications

Any application that supports MCP can connect to Knowledge AI's MCP server to access RAG capabilities.

## Benefits

1. **Standardized Interface**: Tools follow MCP schema standards
2. **Easy Integration**: Simple REST API for tool access
3. **Extensible**: Easy to add new tools
4. **Interoperable**: Works with other MCP-compatible systems

## Future Enhancements

- Full MCP protocol support (stdio transport)
- MCP server discovery
- Tool composition and chaining
- Authentication and authorization for MCP endpoints

