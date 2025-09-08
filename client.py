import os
from openai import OpenAI
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            **kwargs
        )
        return response.choices[0].message.content
    
    def generate_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_logprobs: int = 10,
        **kwargs
    ):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **kwargs
        )
        return response