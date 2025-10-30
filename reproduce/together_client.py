"""
Together AI API client for multi-model inference.
"""

import os
import json
import time
import asyncio
import aiohttp
from typing import List, Dict, Optional

class TogetherClient:
    """Client for Together AI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment or provided")
        
        self.base_url = "https://api.together.xyz/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def generate_sync(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Dict:
        """
        Synchronous generation using Together AI API.
        """
        import requests
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "stop": None,
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
    
    async def generate_async(
        self,
        session: aiohttp.ClientSession,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Dict:
        """
        Asynchronous generation using Together AI API.
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "stop": None,
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def generate_batch_async(
        self,
        model: str,
        batch_messages: List[List[Dict[str, str]]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_concurrent: int = 5,
    ) -> List[Dict]:
        """
        Generate for a batch of messages with concurrency control.
        """
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def generate_with_semaphore(messages):
                async with semaphore:
                    return await self.generate_async(
                        session, model, messages, max_tokens,
                        temperature, top_p, repetition_penalty
                    )
            
            tasks = [generate_with_semaphore(messages) for messages in batch_messages]
            return await asyncio.gather(*tasks, return_exceptions=True)
