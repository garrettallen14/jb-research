"""
Async extensions for OpenAI client to enable parallel reward computation.
"""
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from client import OpenAIClient


class AsyncOpenAIClient(OpenAIClient):
    """Extended OpenAI client with async support for parallel operations."""
    
    def __init__(self, base_url: str, model_name: str, api_key: str = "dummy"):
        super().__init__(base_url, model_name, api_key)
        
        # Async client for parallel operations
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Async version of generate for parallel operations."""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content
    
    async def generate_with_logprobs_async(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_logprobs: int = 10,
        **kwargs
    ):
        """Async version of generate_with_logprobs for parallel operations."""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **kwargs
        )
        return response
