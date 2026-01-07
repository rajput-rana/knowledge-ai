"""Reasoning model support for complex problem-solving."""
from typing import List, Dict, Optional
import httpx
from app.core.config import Settings


class ReasoningModel:
    """Wrapper for reasoning-optimized models (e.g., OpenAI o1)."""
    
    def __init__(self, settings: Settings):
        """
        Initialize reasoning model client.
        
        Args:
            settings: Application settings
        """
        self.api_base = settings.reasoning_api_base
        self.api_key = settings.reasoning_api_key
        self.model = settings.reasoning_model
    
    async def reason(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Use reasoning model to solve a problem.
        
        Args:
            query: User query
            context: Optional context
            system_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with reasoning and answer
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif context:
            messages.append({
                "role": "system",
                "content": f"Use the following context to answer questions:\n\n{context}"
            })
        
        # For reasoning models like o1, the query should be clear and direct
        if context and not system_prompt:
            user_message = f"""Context:
{context}

Question: {query}

Please reason through this step by step and provide a well-justified answer."""
        else:
            user_message = query
        
        messages.append({"role": "user", "content": user_message})
        
        payload = {
            "model": self.model,
            "messages": messages
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            answer = data["choices"][0]["message"]["content"]
            
            return {
                "reasoning": answer,  # For o1, the response includes reasoning
                "answer": answer
            }
    
    def is_available(self) -> bool:
        """Check if reasoning model is configured."""
        return bool(self.api_key and self.api_key != "your-reasoning-api-key-here")


class ReasoningRouter:
    """Routes queries to appropriate model based on complexity."""
    
    def __init__(self, reasoning_model: ReasoningModel, chat_completion, settings: Settings):
        """
        Initialize router.
        
        Args:
            reasoning_model: Reasoning model instance
            chat_completion: Standard chat completion instance
            settings: Application settings
        """
        self.reasoning_model = reasoning_model
        self.chat_completion = chat_completion
        self.settings = settings
    
    async def should_use_reasoning(self, query: str) -> bool:
        """
        Determine if query requires reasoning model.
        
        Args:
            query: User query
            
        Returns:
            True if reasoning model should be used
        """
        if not self.reasoning_model.is_available():
            return False
        
        # Simple heuristic: check for reasoning keywords
        reasoning_keywords = [
            "why", "how", "explain", "analyze", "compare", "evaluate",
            "reasoning", "logic", "deduce", "infer", "conclude"
        ]
        
        query_lower = query.lower()
        has_keywords = any(keyword in query_lower for keyword in reasoning_keywords)
        
        # Check query length/complexity
        is_complex = len(query.split()) > 10
        
        return has_keywords or is_complex or self.settings.always_use_reasoning
    
    async def route(
        self,
        query: str,
        context: Optional[str] = None,
        force_reasoning: bool = False
    ) -> Dict[str, any]:
        """
        Route query to appropriate model.
        
        Args:
            query: User query
            context: Optional context
            force_reasoning: Force use of reasoning model
            
        Returns:
            Response dictionary
        """
        use_reasoning = force_reasoning or await self.should_use_reasoning(query)
        
        if use_reasoning:
            result = await self.reasoning_model.reason(query, context)
            return {
                "answer": result["answer"],
                "reasoning": result.get("reasoning", ""),
                "model_type": "reasoning",
                "model": self.reasoning_model.model
            }
        else:
            # Use standard model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{context}\n\nQuestion: {query}" if context else query}
            ]
            answer = await self.chat_completion.complete(messages)
            return {
                "answer": answer,
                "model_type": "standard",
                "model": self.chat_completion.model
            }

