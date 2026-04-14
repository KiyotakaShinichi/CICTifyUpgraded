import asyncio
from typing import Optional

import aiohttp

from .config import CONFIG, groq_api_key


class GroqClient:
    def __init__(self) -> None:
        self._api_key = groq_api_key()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _session_or_create(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def chat(self, system_prompt: str, user_prompt: str, model: str, *, max_tokens: int | None = None, temperature: float | None = None, timeout_sec: int = 20) -> Optional[str]:
        if not self._api_key:
            return None

        max_tokens = max_tokens or CONFIG.max_tokens
        temperature = CONFIG.temperature if temperature is None else temperature

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        session = await self._session_or_create()
        try:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                return data["choices"][0]["message"]["content"].strip()
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    async def chat_with_fallbacks(self, system_prompt: str, user_prompt: str, *, max_tokens: int | None = None) -> Optional[str]:
        for model in CONFIG.chat_models:
            result = await self.chat(
                system_prompt,
                user_prompt,
                model,
                max_tokens=max_tokens or CONFIG.max_tokens,
                temperature=CONFIG.temperature,
                timeout_sec=CONFIG.request_timeout_sec,
            )
            if result:
                return result
        return None

    async def call_quick_llm(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0) -> Optional[str]:
        return await self.chat(
            "You are a strict classifier. Return concise output only.",
            prompt,
            CONFIG.routing_model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_sec=CONFIG.quick_timeout_sec,
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
