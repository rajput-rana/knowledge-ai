"""LLM chat completion using OpenAI-compatible API."""
from typing import List, Dict, Optional
import httpx
from app.core.config import Settings


class ChatCompletion:
    """OpenAI-compatible chat completion client."""
    
    def __init__(self, settings: Settings):
        """
        Initialize the chat completion client.
        
        Args:
            settings: Application settings
        """
        self.api_base = settings.chat_api_base
        self.api_key = settings.chat_api_key
        self.model = settings.chat_model
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            httpx.HTTPError: If API request fails
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]

