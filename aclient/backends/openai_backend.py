from __future__ import annotations

from typing import Any

from .base import Backend


class OpenAIBackend(Backend):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIBackend requires the 'openai' package. Install with `pip install aclient[openai]`."
            ) from exc

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        request.update(kwargs)

        response = await self._client.chat.completions.create(**request)
        if not response.choices:
            raise ValueError("OpenAI response did not include any choices")

        return self._extract_text(response.choices[0].message.content)

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                item_type = getattr(item, "type", None)
                item_text = getattr(item, "text", None)
                if item_type == "text" and isinstance(item_text, str):
                    parts.append(item_text)
                    continue

                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])

            text = "".join(parts).strip()
            if text:
                return text

        raise ValueError("OpenAI response content could not be normalized into plain text")
