"""Anthropic Messages API adapter (``anthropic.AsyncAnthropic``).

* ``system`` is a top-level parameter
* images are base64 ``image`` content blocks
* JSON: when the request carries a schema, native structured output
  (``output_config.format``) is used; if the model rejects it the adapter
  falls back to prompt-only JSON for the rest of the session.
* The current Messages API has no ``temperature`` parameter, so it is not sent.

Options (``options:`` in llm.yml): ``structured_output`` (bool, default true),
``effort`` (low | medium | high | xhigh | max).
"""

from __future__ import annotations

from typing import Any

import anthropic
from loguru import logger

from tradesense.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponseError,
    LLMTransientError,
    classify_status,
)
from tradesense.llm.factory import register_provider
from tradesense.llm.messages import ImagePart, LLMRequest, TextPart


@register_provider("anthropic")
class AnthropicProvider(LLMProvider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: anthropic.AsyncAnthropic | None = None
        self._structured_ok = bool(self.option("structured_output", True))

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.config.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    # ------------------------------------------------------------------ translation

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        messages = []
        for message in request.messages:
            blocks: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.media_type,
                                "data": part.b64,
                            },
                        }
                    )
            messages.append({"role": message.role, "content": blocks})
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens_for(request),
            "messages": messages,
        }
        if request.system:
            params["system"] = request.system
        output_config: dict[str, Any] = {}
        if self.json_mode_enabled(request) and request.json_schema and self._structured_ok:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        effort = self.option("effort")
        if effort:
            output_config["effort"] = effort
        if output_config:
            params["output_config"] = output_config
        return params

    # ------------------------------------------------------------------ calls

    async def _generate(self, request: LLMRequest) -> tuple[str, int, int]:
        params = self.build_params(request)
        try:
            message = await self.client.messages.create(**params)
        except anthropic.BadRequestError as exc:
            fmt = params.get("output_config", {}).get("format")
            if fmt and self._structured_ok and _mentions_format(exc):
                logger.warning(
                    "{}: structured output rejected ({}); using prompt-only JSON",
                    self.name,
                    _short(exc),
                )
                self._structured_ok = False
                return await self._generate(request)
            raise self._map(exc) from exc
        except anthropic.APIError as exc:
            raise self._map(exc) from exc

        text = "".join(
            getattr(block, "text", "") for block in message.content if block.type == "text"
        )
        if not text.strip():
            raise LLMResponseError(
                f"{self.name}: empty response (stop_reason={message.stop_reason})",
                provider=self.name,
            )
        usage = message.usage
        return text, int(usage.input_tokens or 0), int(usage.output_tokens or 0)

    def _map(self, exc: Exception) -> LLMError:
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMTransientError(f"{self.name}: request timed out", provider=self.name)
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMTransientError(f"{self.name}: connection error", provider=self.name)
        if isinstance(exc, anthropic.APIStatusError):
            return classify_status(exc.status_code, f"{self.name}: {_short(exc)}", self.name)
        return LLMError(f"{self.name}: {_short(exc)}", provider=self.name)

    async def health_check(self) -> bool:
        await self.client.models.list(limit=1)
        return True

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _short(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:300]


def _mentions_format(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(word in text for word in ("output_config", "format", "schema", "structured"))
