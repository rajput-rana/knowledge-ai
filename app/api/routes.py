"""FastAPI route handlers."""
from typing import Annotated, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import numpy as np

from app.api.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, SourceDocument
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.embeddings.embedder import OpenAIEmbedder
from app.vector_store.faiss_store import FAISSVectorStore
from app.retrieval.retriever import Retriever
from app.llm.chat import ChatCompletion
from app.llm.reasoning import ReasoningModel, ReasoningRouter
from app.rag.pipeline import RAGPipeline
from app.agents.rag_agent import RAGAgent
from app.agents.tools import RAGRetrievalTool, DocumentIngestionTool, QueryRefinementTool
from app.prompts.templates import PromptStyle
from app.mcp.server import RAGMCPServer
from app.mcp.client import MCPClient, MCPToolRegistry
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging

logger = setup_logging()

router = APIRouter()


class VectorStoreManager:
    """Manages shared vector store instance for the application."""
    _instance: Optional[FAISSVectorStore] = None
    
    @classmethod
    def get_instance(cls, settings: Settings) -> FAISSVectorStore:
        """Get or create the shared vector store instance."""
        if cls._instance is None:
            cls._instance = FAISSVectorStore(
                dimension=settings.vector_dimension,
                index_path=settings.faiss_index_path
            )
        return cls._instance


def get_vector_store(settings: Annotated[Settings, Depends(get_settings)]) -> FAISSVectorStore:
    """
    Dependency to get vector store instance.
    Returns a shared instance to persist data across requests.
    """
    return VectorStoreManager.get_instance(settings)


def get_embedder(settings: Annotated[Settings, Depends(get_settings)]) -> OpenAIEmbedder:
    """Dependency to get embedder instance."""
    return OpenAIEmbedder(settings)


def get_chat_completion(settings: Annotated[Settings, Depends(get_settings)]) -> ChatCompletion:
    """Dependency to get chat completion instance."""
    return ChatCompletion(settings)


def get_retriever(
    embedder: Annotated[OpenAIEmbedder, Depends(get_embedder)],
    vector_store: Annotated[FAISSVectorStore, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> Retriever:
    """Dependency to get retriever instance."""
    return Retriever(embedder, vector_store, top_k=settings.top_k)


def get_rag_pipeline(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    chat_completion: Annotated[ChatCompletion, Depends(get_chat_completion)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> RAGPipeline:
    """Dependency to get RAG pipeline instance."""
    return RAGPipeline(retriever, chat_completion, settings)


def get_reasoning_model(settings: Annotated[Settings, Depends(get_settings)]) -> ReasoningModel:
    """Dependency to get reasoning model instance."""
    return ReasoningModel(settings)


def get_reasoning_router(
    reasoning_model: Annotated[ReasoningModel, Depends(get_reasoning_model)],
    chat_completion: Annotated[ChatCompletion, Depends(get_chat_completion)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> ReasoningRouter:
    """Dependency to get reasoning router instance."""
    return ReasoningRouter(reasoning_model, chat_completion, settings)


def get_rag_agent(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    chat_completion: Annotated[ChatCompletion, Depends(get_chat_completion)],
    settings: Annotated[Settings, Depends(get_settings)],
    loader: Annotated[DocumentLoader, Depends(lambda: DocumentLoader())]
) -> RAGAgent:
    """Dependency to get RAG agent instance."""
    tools = [
        RAGRetrievalTool(retriever),
        DocumentIngestionTool(loader),
        QueryRefinementTool()
    ]
    return RAGAgent(tools, chat_completion, settings, max_iterations=settings.agent_max_iterations)


def get_mcp_server(
    rag_pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
    embedder: Annotated[OpenAIEmbedder, Depends(get_embedder)],
    vector_store: Annotated[FAISSVectorStore, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> Optional[RAGMCPServer]:
    """Dependency to get MCP server instance."""
    if not settings.enable_mcp_server:
        return None
    
    loader = DocumentLoader()
    return RAGMCPServer(
        rag_pipeline=rag_pipeline,
        retriever=retriever,
        embedder=embedder,
        vector_store=vector_store,
        loader=loader,
        settings=settings
    )


def get_mcp_registry() -> MCPToolRegistry:
    """Dependency to get MCP tool registry."""
    return MCPToolRegistry()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    vector_store: Annotated[FAISSVectorStore, Depends(get_vector_store)],
    embedder: Annotated[OpenAIEmbedder, Depends(get_embedder)]
):
    """
    Ingest a document into the knowledge base.
    
    Args:
        request: Ingest request with text and optional metadata
        settings: Application settings
        vector_store: Vector store instance
        embedder: Embedder instance
        
    Returns:
        Ingest response with document ID and chunk count
    """
    try:
        # Load document
        loader = DocumentLoader()
        document = loader.load_text(
            text=request.text,
            doc_id=request.doc_id,
            metadata=request.metadata or {}
        )
        
        # Chunk document
        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = chunker.chunk_document(document)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created from document")
        
        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await embedder.embed_batch(chunk_texts)
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Add to vector store
        vector_store.add(vectors, chunks)
        
        # Save index if path is configured
        if settings.faiss_index_path:
            vector_store.save()
        
        logger.info(f"Ingested document {document.id} with {len(chunks)} chunks")
        
        return IngestResponse(
            doc_id=document.id,
            chunks_created=len(chunks),
            message=f"Successfully ingested document with {len(chunks)} chunks"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    rag_agent: Annotated[RAGAgent, Depends(get_rag_agent)],
    reasoning_router: Annotated[ReasoningRouter, Depends(get_reasoning_router)],
    settings: Annotated[Settings, Depends(get_settings)]
):
    """
    Query the knowledge base with support for agentic AI and reasoning models.
    
    Args:
        request: Query request with user question and options
        rag_pipeline: RAG pipeline instance
        rag_agent: RAG agent instance
        reasoning_router: Reasoning router instance
        settings: Application settings
        
    Returns:
        Query response with answer and sources
    """
    try:
        # Determine which mode to use
        use_agent = request.use_agent or (settings.enable_agentic_mode and request.use_agent is None)
        use_reasoning = request.use_reasoning
        
        answer = None
        source_documents = []
        reasoning = None
        agent_trace = None
        model_type = "standard"
        
        # Override prompt style if specified
        original_style = settings.prompt_style
        if request.prompt_style:
            settings.prompt_style = request.prompt_style
        
        try:
            if use_agent:
                # Use agentic mode
                logger.info(f"Using agentic mode for query: {request.query}")
                agent_state = await rag_agent.run(request.query)
                answer = agent_state.final_answer or "I couldn't generate an answer."
                agent_trace = agent_state.to_dict()
                model_type = "agentic"
                
                # Extract sources from agent actions
                for action in agent_state.actions:
                    if action["tool"] == "rag_retrieve":
                        # Get sources from retrieval - need to get retriever from pipeline
                        retriever = rag_pipeline.retriever
                        retrieved_chunks = await retriever.retrieve(request.query)
                        source_documents = [doc for doc, _ in retrieved_chunks]
            
            elif use_reasoning and reasoning_router.reasoning_model.is_available():
                # Use reasoning model
                logger.info(f"Using reasoning model for query: {request.query}")
                retrieved_chunks = await rag_pipeline.retriever.retrieve(request.query)
                context = rag_pipeline._build_context(retrieved_chunks) if retrieved_chunks else None
                
                result = await reasoning_router.route(request.query, context, force_reasoning=True)
                answer = result["answer"]
                reasoning = result.get("reasoning")
                model_type = "reasoning"
                
                source_documents = [doc for doc, _ in retrieved_chunks] if retrieved_chunks else []
            
            else:
                # Standard RAG mode
                answer, source_documents = await rag_pipeline.query(request.query)
                model_type = "standard"
        
        finally:
            # Restore original prompt style
            settings.prompt_style = original_style
        
        # Format source documents
        sources = [
            SourceDocument(
                doc_id=doc.id,
                chunk_index=doc.metadata.get("chunk_index"),
                text_preview=doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                metadata=doc.metadata
            )
            for doc in source_documents
        ]
        
        return QueryResponse(
            answer=answer or "I couldn't generate an answer.",
            sources=sources,
            num_sources=len(sources),
            model_type=model_type,
            reasoning=reasoning,
            agent_trace=agent_trace
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/mcp/tools")
async def list_mcp_tools(
    mcp_server: Annotated[Optional[RAGMCPServer], Depends(get_mcp_server)]
):
    """
    List available MCP tools.
    
    Returns:
        List of MCP tools exposed by the server
    """
    if not mcp_server:
        raise HTTPException(
            status_code=503,
            detail="MCP server is not enabled"
        )
    
    try:
        tools = await mcp_server.list_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in tools
            ]
        }
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")


@router.post("/mcp/tools/{tool_name}")
async def call_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    mcp_server: Annotated[Optional[RAGMCPServer], Depends(get_mcp_server)]
):
    """
    Call an MCP tool.
    
    Args:
        tool_name: Name of the tool to call
        arguments: Tool arguments
        
    Returns:
        Tool execution result
    """
    if not mcp_server:
        raise HTTPException(
            status_code=503,
            detail="MCP server is not enabled"
        )
    
    try:
        result = await mcp_server.call_tool(tool_name, arguments)
        
        # Extract text content
        texts = []
        for content in result:
            if hasattr(content, 'text'):
                texts.append(content.text)
            elif isinstance(content, dict) and 'text' in content:
                texts.append(content['text'])
        
        return {
            "tool": tool_name,
            "result": "\n".join(texts)
        }
    
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calling tool: {str(e)}")

