"""MCP-compatible server implementation for exposing RAG capabilities."""
from typing import List, Dict, Any, Optional
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.embeddings.embedder import OpenAIEmbedder
from app.vector_store.faiss_store import FAISSVectorStore
from app.core.config import Settings
import numpy as np


class MCPTool:
    """MCP tool definition."""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class MCPTextContent:
    """MCP text content."""
    
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class RAGMCPServer:
    """MCP-compatible server that exposes RAG capabilities as tools."""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        retriever: Retriever,
        embedder: OpenAIEmbedder,
        vector_store: FAISSVectorStore,
        loader: DocumentLoader,
        settings: Settings
    ):
        """
        Initialize MCP server.
        
        Args:
            rag_pipeline: RAG pipeline instance
            retriever: Retriever instance
            embedder: Embedder instance
            vector_store: Vector store instance
            loader: Document loader instance
            settings: Application settings
        """
        self.rag_pipeline = rag_pipeline
        self.retriever = retriever
        self.embedder = embedder
        self.vector_store = vector_store
        self.loader = loader
        self.settings = settings
    
    async def list_tools(self) -> List[MCPTool]:
        """List available MCP tools."""
        return [
            MCPTool(
                name="query_knowledge_base",
                description="Query the knowledge base using RAG. Retrieves relevant documents and generates an answer.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to ask"
                        },
                        "prompt_style": {
                            "type": "string",
                            "enum": ["standard", "chain_of_thought", "few_shot", "reasoning"],
                            "description": "Prompt style to use",
                            "default": "standard"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of documents to retrieve",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        }
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="search_knowledge_base",
                description="Search the knowledge base for relevant documents without generating an answer.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        }
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="ingest_document",
                description="Add a document to the knowledge base. The document will be chunked, embedded, and stored.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Document text content"
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Optional document ID"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata dictionary"
                        }
                    },
                    "required": ["text"]
                }
            ),
            MCPTool(
                name="get_knowledge_base_stats",
                description="Get statistics about the knowledge base (number of documents, chunks, etc.).",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[MCPTextContent]:
            """Handle tool calls."""
            
            if name == "query_knowledge_base":
                query = arguments.get("query", "")
                prompt_style = arguments.get("prompt_style", "standard")
                top_k = arguments.get("top_k", self.settings.top_k)
                
                if not query:
                    return [MCPTextContent(
                        text="Error: query parameter is required"
                    )]
                
                # Temporarily override top_k
                original_top_k = self.settings.top_k
                self.settings.top_k = top_k
                
                try:
                    # Override prompt style temporarily
                    original_style = self.settings.prompt_style
                    self.settings.prompt_style = prompt_style
                    
                    answer, sources = await self.rag_pipeline.query(query)
                    
                    # Restore original style
                    self.settings.prompt_style = original_style
                    
                    # Format response
                    response = f"Answer: {answer}\n\n"
                    if sources:
                        response += f"Sources ({len(sources)}):\n"
                        for i, doc in enumerate(sources[:top_k], 1):
                            response += f"{i}. {doc.id}\n"
                            if doc.metadata.get("chunk_index") is not None:
                                response += f"   Chunk {doc.metadata['chunk_index']}\n"
                    
                    return [MCPTextContent(text=response)]
                
                finally:
                    self.settings.top_k = original_top_k
            
            elif name == "search_knowledge_base":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", self.settings.top_k)
                
                if not query:
                    return [MCPTextContent(
                        text="Error: query parameter is required"
                    )]
                
                # Temporarily override top_k
                original_top_k = self.settings.top_k
                self.settings.top_k = top_k
                
                try:
                    results = await self.retriever.retrieve(query)
                    
                    if not results:
                        return [MCPTextContent(
                            text="No documents found matching the query."
                        )]
                    
                    response = f"Found {len(results)} relevant documents:\n\n"
                    for i, (doc, score) in enumerate(results[:top_k], 1):
                        response += f"{i}. Document ID: {doc.id}\n"
                        response += f"   Similarity: {score:.3f}\n"
                        response += f"   Preview: {doc.text[:200]}...\n\n"
                    
                    return [MCPTextContent(text=response)]
                
                finally:
                    self.settings.top_k = original_top_k
            
            elif name == "ingest_document":
                text = arguments.get("text", "")
                doc_id = arguments.get("doc_id")
                metadata = arguments.get("metadata", {})
                
                if not text:
                    return [MCPTextContent(
                        text="Error: text parameter is required"
                    )]
                
                try:
                    # Load document
                    document = self.loader.load_text(
                        text=text,
                        doc_id=doc_id,
                        metadata=metadata
                    )
                    
                    # Chunk document
                    from app.ingestion.chunker import TextChunker
                    chunker = TextChunker(
                        chunk_size=self.settings.chunk_size,
                        chunk_overlap=self.settings.chunk_overlap
                    )
                    chunks = chunker.chunk_document(document)
                    
                    if not chunks:
                        return [MCPTextContent(
                            text="Error: Failed to chunk document"
                        )]
                    
                    # Generate embeddings
                    chunk_texts = [chunk.text for chunk in chunks]
                    embeddings = await self.embedder.embed_batch(chunk_texts)
                    
                    # Convert to numpy array
                    vectors = np.array(embeddings, dtype=np.float32)
                    
                    # Add to vector store
                    self.vector_store.add(vectors, chunks)
                    
                    # Save index if path is configured
                    if self.settings.faiss_index_path:
                        self.vector_store.save()
                    
                    return [MCPTextContent(
                        text=f"Successfully ingested document '{document.id}' with {len(chunks)} chunks."
                    )]
                
                except Exception as e:
                    return [MCPTextContent(
                        text=f"Error ingesting document: {str(e)}"
                    )]
            
            elif name == "get_knowledge_base_stats":
                total_docs = len(self.vector_store)
                
                # Count unique parent documents
                parent_docs = set()
                for doc in self.vector_store.documents:
                    parent_id = doc.metadata.get("parent_doc_id", doc.id)
                    parent_docs.add(parent_id)
                
                stats = f"Knowledge Base Statistics:\n"
                stats += f"- Total chunks: {total_docs}\n"
                stats += f"- Unique documents: {len(parent_docs)}\n"
                stats += f"- Vector dimension: {self.settings.vector_dimension}\n"
                stats += f"- Chunk size: {self.settings.chunk_size}\n"
                stats += f"- Chunk overlap: {self.settings.chunk_overlap}\n"
                
                return [MCPTextContent(text=stats)]
            
            else:
                return [MCPTextContent(
                    text=f"Unknown tool: {name}"
                )]

