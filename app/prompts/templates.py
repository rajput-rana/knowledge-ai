"""Prompt templates and few-shot examples for enhanced RAG."""
from typing import List, Dict, Optional
from enum import Enum


class PromptStyle(str, Enum):
    """Prompt style options."""
    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    REASONING = "reasoning"


class PromptTemplate:
    """Enhanced prompt template system."""
    
    STANDARD_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain enough information to answer the question, say so. 
Cite specific document IDs when referencing information from the context."""

    CHAIN_OF_THOUGHT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context. 
When answering, think step by step:
1. First, identify what information is needed to answer the question
2. Then, find relevant information in the provided context
3. Analyze and synthesize the information
4. Finally, provide a clear answer with citations

If the context doesn't contain enough information, explain what's missing and why you cannot answer completely."""

    REASONING_SYSTEM_PROMPT = """You are an expert reasoning assistant. When answering questions based on the provided context:
1. Break down complex questions into sub-questions
2. Analyze each piece of information systematically
3. Draw logical conclusions from the evidence
4. Consider alternative interpretations
5. Provide a well-reasoned answer with clear justification

Always cite your sources using document IDs."""

    FEW_SHOT_EXAMPLES = [
        {
            "query": "What are the main features of FastAPI?",
            "context": "[Document ID: doc1_chunk_0]\nFastAPI is a modern web framework with automatic API documentation, type validation, and async support.\n---\n\n",
            "answer": "Based on the context, FastAPI has several main features:\n1. Automatic API documentation\n2. Type validation\n3. Async support\n\nSource: doc1_chunk_0"
        },
        {
            "query": "How does RAG work?",
            "context": "[Document ID: doc2_chunk_1]\nRAG combines retrieval of relevant documents with generation of answers using LLMs.\n---\n\n",
            "answer": "RAG (Retrieval-Augmented Generation) works by first retrieving relevant documents from a knowledge base, then using those documents as context for a language model to generate accurate answers.\n\nSource: doc2_chunk_1"
        }
    ]
    
    @classmethod
    def build_prompt(
        cls,
        query: str,
        context: str,
        style: PromptStyle = PromptStyle.STANDARD,
        include_examples: bool = False,
        custom_instructions: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build enhanced prompt based on style.
        
        Args:
            query: User query
            context: Retrieved context
            style: Prompt style to use
            include_examples: Whether to include few-shot examples
            custom_instructions: Optional custom instructions
            
        Returns:
            List of message dictionaries for chat completion
        """
        # Select system prompt based on style
        if style == PromptStyle.CHAIN_OF_THOUGHT:
            system_prompt = cls.CHAIN_OF_THOUGHT_SYSTEM_PROMPT
        elif style == PromptStyle.REASONING:
            system_prompt = cls.REASONING_SYSTEM_PROMPT
        else:
            system_prompt = cls.STANDARD_SYSTEM_PROMPT
        
        # Add custom instructions if provided
        if custom_instructions:
            system_prompt += f"\n\nAdditional instructions: {custom_instructions}"
        
        # Build user prompt
        if style == PromptStyle.CHAIN_OF_THOUGHT:
            user_prompt = cls._build_chain_of_thought_prompt(query, context)
        elif include_examples and style == PromptStyle.FEW_SHOT:
            user_prompt = cls._build_few_shot_prompt(query, context)
        else:
            user_prompt = cls._build_standard_prompt(query, context)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @classmethod
    def _build_standard_prompt(cls, query: str, context: str) -> str:
        """Build standard prompt."""
        return f"""Context:
{context}

Question: {query}

Answer:"""
    
    @classmethod
    def _build_chain_of_thought_prompt(cls, query: str, context: str) -> str:
        """Build chain-of-thought prompt."""
        return f"""Context:
{context}

Question: {query}

Let's think step by step:

1. What information is needed to answer this question?
2. What relevant information is available in the context?
3. How can we synthesize this information?
4. What is the final answer?

Answer:"""
    
    @classmethod
    def _build_few_shot_prompt(cls, query: str, context: str) -> str:
        """Build few-shot prompt with examples."""
        examples_text = "\n\n".join([
            f"Example {i+1}:\nQuestion: {ex['query']}\nContext: {ex['context']}\nAnswer: {ex['answer']}"
            for i, ex in enumerate(cls.FEW_SHOT_EXAMPLES)
        ])
        
        return f"""Here are some examples of how to answer questions based on context:

{examples_text}

Now answer this question:

Context:
{context}

Question: {query}

Answer:"""

