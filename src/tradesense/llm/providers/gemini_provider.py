"""Google Gemini adapter (``google-genai`` async client).

* system prompt -> ``system_instruction``
* images -> inline byte parts (``Part.from_bytes``)
* JSON -> ``response_mime_type="application/json"``

Options: ``thinking_budget`` (int; e.g. 0 disables thinking on Flash models so the
output budget is not consumed by thoughts).
"""

from __future__ import annotations

from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from tradesense.llm.base import (
    LLMProvider,
    LLMResponseError,
    LLMTransientError,
    classify_status,
)
from tradesense.llm.factory import register_provider
from tradesense.llm.messages import ImagePart, LLMRequest, TextPart


@register_provider("gemini")
class GeminiProvider(LLMProvider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
        return self._client

    # ------------------------------------------------------------------ translation

    def build_request(
        self, request: LLMRequest
    ) -> tuple[list[types.ContentUnion], types.GenerateContentConfig]:
        contents: list[types.ContentUnion] = []
        for message in request.messages:
            parts: list[types.Part] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append(types.Part.from_text(text=part.text))
                elif isinstance(part, ImagePart):
                    parts.append(types.Part.from_bytes(data=part.data, mime_type=part.media_type))
            role = "model" if message.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=parts))

        config: dict[str, Any] = {
            "temperature": self.temperature_for(request),
            "max_output_tokens": self.max_tokens_for(request),
        }
        if request.system:
            config["system_instruction"] = request.system
        if self.json_mode_enabled(request):
            config["response_mime_type"] = "application/json"
        budget = self.option("thinking_budget")
        if isinstance(budget, int):
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
        return contents, types.GenerateContentConfig(**config)

    # ------------------------------------------------------------------ calls

    async def _generate(self, request: LLMRequest) -> tuple[str, int, int]:
        contents, config = self.build_request(request)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except genai_errors.APIError as exc:
            raise classify_status(exc.code, f"{self.name}: {str(exc)[:300]}", self.name) from exc
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise LLMTransientError(f"{self.name}: request timed out", provider=self.name) from exc
        except (httpx.TransportError, ConnectionError) as exc:
            raise LLMTransientError(f"{self.name}: connection error", provider=self.name) from exc

        text = response.text or ""
        if not text.strip():
            raise LLMResponseError(f"{self.name}: empty response", provider=self.name)
        usage = response.usage_metadata
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
            getattr(usage, "thoughts_token_count", 0) or 0
        )
        return text, input_tokens, output_tokens

    async def health_check(self) -> bool:
        await self.client.aio.models.get(model=self.model)
        return True
