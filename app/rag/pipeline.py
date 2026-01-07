"""RAG pipeline combining retrieval and generation."""
from typing import List, Tuple, Dict, Optional
from app.retrieval.retriever import Retriever
from app.llm.chat import ChatCompletion
from app.ingestion.loader import Document
from app.core.config import Settings
from app.prompts.templates import PromptTemplate, PromptStyle


class RAGPipeline:
    """RAG pipeline that retrieves context and generates answers."""
    
    def __init__(
        self,
        retriever: Retriever,
        chat_completion: ChatCompletion,
        settings: Settings
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Retriever instance
            chat_completion: Chat completion instance
            settings: Application settings
        """
        self._retriever = retriever
        self.chat_completion = chat_completion
        self.settings = settings
        self.max_context_length = settings.max_context_length
    
    @property
    def retriever(self):
        """Expose retriever for external access."""
        return self._retriever
    
    def _build_context(self, retrieved_chunks: List[Tuple[Document, float]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            retrieved_chunks: List of (Document, similarity_score) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for doc, score in retrieved_chunks:
            chunk_text = f"[Document ID: {doc.id}]\n{doc.text}\n"
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > self.max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += chunk_length
        
        return "\n---\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        Build the RAG prompt with context using enhanced templates.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            List of message dictionaries for chat completion
        """
        # Determine prompt style from settings
        style_map = {
            "standard": PromptStyle.STANDARD,
            "chain_of_thought": PromptStyle.CHAIN_OF_THOUGHT,
            "few_shot": PromptStyle.FEW_SHOT,
            "reasoning": PromptStyle.REASONING
        }
        
        style = style_map.get(self.settings.prompt_style, PromptStyle.STANDARD)
        
        return PromptTemplate.build_prompt(
            query=query,
            context=context,
            style=style,
            include_examples=self.settings.include_few_shot_examples
        )
    
    async def query(self, query: str) -> Tuple[str, List[Document]]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (answer, source_documents)
        """
        # Retrieve relevant chunks
        retrieved_chunks = await self._retriever.retrieve(query)
        
        if not retrieved_chunks:
            return (
                "I couldn't find any relevant information in the knowledge base.",
                []
            )
        
        # Build context
        context = self._build_context(retrieved_chunks)
        
        # Build prompt
        messages = self._build_prompt(query, context)
        
        # Generate answer
        answer = await self.chat_completion.complete(messages, temperature=0.7)
        
        # Extract source documents
        source_documents = [doc for doc, _ in retrieved_chunks]
        
        return answer, source_documents

