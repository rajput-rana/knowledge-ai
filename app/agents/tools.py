"""Built-in agent tools."""
from typing import Dict, Any, List
from app.agents.base import Tool
from app.retrieval.retriever import Retriever
from app.ingestion.loader import DocumentLoader, Document


class RAGRetrievalTool(Tool):
    """Tool for retrieving information from knowledge base."""
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
    
    @property
    def name(self) -> str:
        return "rag_retrieve"
    
    @property
    def description(self) -> str:
        return "Retrieve relevant information from the knowledge base using semantic search. Use this when you need to find information to answer a question."
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        if not query:
            return {"error": "Query parameter is required"}
        
        results = await self.retriever.retrieve(query)
        
        return {
            "results": [
                {
                    "doc_id": doc.id,
                    "text": doc.text,
                    "similarity": float(score),
                    "metadata": doc.metadata
                }
                for doc, score in results
            ],
            "count": len(results)
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant information"
                }
            },
            "required": ["query"]
        }


class DocumentIngestionTool(Tool):
    """Tool for ingesting new documents."""
    
    def __init__(self, loader: DocumentLoader):
        self.loader = loader
    
    @property
    def name(self) -> str:
        return "ingest_document"
    
    @property
    def description(self) -> str:
        return "Add a new document to the knowledge base. Use this when you need to store new information."
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data.get("text", "")
        if not text:
            return {"error": "Text parameter is required"}
        
        doc_id = input_data.get("doc_id")
        metadata = input_data.get("metadata", {})
        
        document = self.loader.load_text(text, doc_id=doc_id, metadata=metadata)
        
        return {
            "doc_id": document.id,
            "message": "Document loaded successfully (embedding and storage handled separately)"
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
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
                    "description": "Optional metadata"
                }
            },
            "required": ["text"]
        }


class QueryRefinementTool(Tool):
    """Tool for refining and expanding queries."""
    
    @property
    def name(self) -> str:
        return "refine_query"
    
    @property
    def description(self) -> str:
        return "Refine or expand a query to improve search results. Use this when initial search doesn't yield good results."
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        original_query = input_data.get("query", "")
        refinement_type = input_data.get("type", "expand")  # expand, narrow, rephrase
        
        # Simple query refinement logic
        # In production, this could use an LLM to refine queries
        if refinement_type == "expand":
            refined = f"{original_query} related information details"
        elif refinement_type == "narrow":
            refined = f"{original_query} specific"
        else:
            refined = original_query
        
        return {
            "original_query": original_query,
            "refined_query": refined,
            "refinement_type": refinement_type
        }
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to refine"
                },
                "type": {
                    "type": "string",
                    "enum": ["expand", "narrow", "rephrase"],
                    "description": "Type of refinement"
                }
            },
            "required": ["query"]
        }

