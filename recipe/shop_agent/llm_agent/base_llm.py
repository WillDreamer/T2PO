# recipe/shop_agent/llm_agent/base_llm.py
# 只保留 OpenAI Provider 和 ConcurrentLLM
# DeepSeek, Anthropic, Together 先注释掉，方便未来扩展
from dotenv import load_dotenv
load_dotenv(override=True)
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Union
import os
import asyncio
import time
from openai import AsyncOpenAI


# ===================== 基础结构 =====================

@dataclass
class LLMResponse:
    """Unified response format across all LLM providers"""
    content: str
    model_name: str


class LLMProvider:
    """Minimal abstract class (for interface consistency)"""
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        raise NotImplementedError


# ===================== ✅ OpenAI Provider =====================

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or missing in environment variables")

        self.client = AsyncOpenAI(api_key=self.api_key)
        print(f"[DEBUG] Using OpenAI model: {self.model_name}")
        print(f"[DEBUG] OPENAI_API_KEY prefix: {str(self.api_key)[:16] if self.api_key else None}")


    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        text = response.choices[0].message.content
        return LLMResponse(content=text, model_name=response.model)


# ===================== ✅ DeepSeek Provider =====================
class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider implementation (OpenAI-compatible SDK)"""
    def __init__(self, model_name: str = "deepseek-chat", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided or missing in environment variables")

        # DeepSeek uses OpenAI-compatible endpoint but different base_url
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        print(f"[DEBUG] Using DeepSeek model: {self.model_name}")
        print(f"[DEBUG] DEEPSEEK_API_KEY prefix: {str(self.api_key)[:16] if self.api_key else None}")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        text = response.choices[0].message.content
        return LLMResponse(content=text, model_name=response.model)




# ===================== ⚙️ ConcurrentLLM =====================

class ConcurrentLLM:
    """Unified concurrent interface for multiple providers (OpenAI + DeepSeek)"""

    def __init__(
        self,
        provider: Union[str, LLMProvider] = "openai",
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_concurrency: int = 4
    ):
        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            p = provider.lower()
            if p == "openai":
                self.provider = OpenAIProvider(model_name, api_key)
            elif p == "deepseek":
                self.provider = DeepSeekProvider(model_name, api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        self.max_concurrency = max_concurrency
        self._semaphore = None

    @property
    def semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        async with self.semaphore:
            return await self.provider.generate(messages, **kwargs)

    def run_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        """Batch inference with retry"""
        results = [None] * len(messages_list)
        position_map = {id(m): i for i, m in enumerate(messages_list)}

        current_batch = messages_list.copy()
        retry_count = 0
        max_retries = kwargs.get("max_retries", 3)

        while current_batch and retry_count < max_retries:
            async def process_batch():
                self._semaphore = None
                batch_results, failures = [], []
                tasks = [(m, asyncio.create_task(self.generate(m, **kwargs)))
                         for m in current_batch]
                for messages, task in tasks:
                    try:
                        resp = await task
                        idx = position_map[id(messages)]
                        batch_results.append((idx, {
                            "messages": messages,
                            "response": resp.content,
                            "model": resp.model_name,
                            "success": True
                        }))
                    except Exception as e:
                        print(f"[DEBUG] API error: {e}")
                        traceback.print_exc()
                        failures.append(messages)
                return batch_results, failures

            batch_results, next_batch = asyncio.run(process_batch())
            for idx, res in batch_results:
                results[idx] = res

            if next_batch:
                retry_count += 1
                print(f"[WARN] {len(next_batch)} failed, retry {retry_count}/{max_retries}")
                current_batch = next_batch
                time.sleep(2)
            else:
                current_batch = []
                break

        return results, current_batch


# ===================== ✅ 简单测试 =====================
# if __name__ == "__main__":
#     llm = ConcurrentLLM(provider="openai", model_name="gpt-4o-mini", max_concurrency=3)
#     messages = [
#         [{"role": "user", "content": "Say 1"}],
#         [{"role": "user", "content": "Say 2"}],
#         [{"role": "user", "content": "Say 3"}],
#     ]
#     results, failed = llm.run_batch(messages, temperature=0.7)
#     for r in results:
#         if r:
#             print(r["response"])