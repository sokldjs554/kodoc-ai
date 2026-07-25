"""OpenAI 호환 Chat Completions 클라이언트.

vLLM(`vllm serve ...`)이 OpenAI 호환 API를 제공하므로, 서비스 코드는 특정
추론 엔진에 묶이지 않는다. base_url만 바꾸면 vLLM ↔ TGI ↔ 상용 API 사이를
오갈 수 있고, 벤치마크에서 엔진 간 비교도 같은 코드로 수행한다.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, messages: list[dict], **kwargs: Any) -> dict:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        return payload

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = await self._client.post(
            "/chat/completions",
            json=self._payload(messages, temperature=temperature, max_tokens=max_tokens),
        )
        if response.status_code != 200:
            raise LLMError(f"LLM 서버 오류 {response.status_code}: {response.text[:500]}")
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """SSE 스트리밍으로 토큰 델타를 순서대로 낸다."""
        payload = self._payload(
            messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise LLMError(f"LLM 서버 오류 {response.status_code}: {body[:500]!r}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
