"""Ollama adapter (``ollama.AsyncClient``). Models are assumed to be pulled already.

* system prompt is the first message
* images are a base64 list on the message
* JSON -> ``format="json"``

Options: ``think`` (bool / "low" | "medium" | "high": controls reasoning on thinking
models; ``false`` keeps responses fast), ``num_ctx`` (context window),
``num_gpu`` (layers offloaded to the GPU; ``0`` = CPU only), ``num_thread``,
``keep_alive`` (e.g. "10m").
"""

from __future__ import annotations

from typing import Any

import httpx
import ollama

from tradesense.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponseError,
    LLMTransientError,
    classify_status,
)
from tradesense.llm.factory import register_provider
from tradesense.llm.messages import LLMRequest

DEFAULT_HOST = "http://localhost:11434"


@register_provider("ollama")
class OllamaProvider(LLMProvider):
    requires_api_key = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: ollama.AsyncClient | None = None

    @property
    def is_local(self) -> bool:
        return True

    @property
    def host(self) -> str:
        return self.config.base_url or DEFAULT_HOST

    @property
    def client(self) -> ollama.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            self._client = ollama.AsyncClient(host=self.host, timeout=self.timeout, headers=headers)
        return self._client

    # ------------------------------------------------------------------ translation

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            entry: dict[str, Any] = {"role": message.role, "content": message.text}
            if message.images:
                entry["images"] = [image.b64 for image in message.images]
            messages.append(entry)
        options: dict[str, Any] = {
            "temperature": self.temperature_for(request),
            "num_predict": self.max_tokens_for(request),
        }
        for key in ("num_ctx", "num_gpu", "num_thread"):
            value = self.option(key)
            if isinstance(value, int) and not isinstance(value, bool):
                options[key] = value
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self.json_mode_enabled(request):
            params["format"] = "json"
        think = self.option("think")
        if think is not None:
            params["think"] = think
        keep_alive = self.option("keep_alive")
        if keep_alive is not None:
            params["keep_alive"] = keep_alive
        return params

    # ------------------------------------------------------------------ calls

    async def _generate(self, request: LLMRequest) -> tuple[str, int, int]:
        params = self.build_params(request)
        try:
            response = await self.client.chat(**params)
        except ollama.ResponseError as exc:
            raise classify_status(exc.status_code, f"{self.name}: {exc.error}", self.name) from exc
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise LLMTransientError(f"{self.name}: request timed out", provider=self.name) from exc
        except (httpx.TransportError, ConnectionError) as exc:
            raise LLMTransientError(
                f"{self.name}: cannot reach Ollama at {self.host}", provider=self.name
            ) from exc
        except ollama.RequestError as exc:
            raise LLMError(f"{self.name}: {exc}", provider=self.name) from exc

        text = response.message.content or ""
        if not text.strip():
            raise LLMResponseError(
                f"{self.name}: empty response (done_reason={response.done_reason})",
                provider=self.name,
            )
        return text, int(response.prompt_eval_count or 0), int(response.eval_count or 0)

    async def health_check(self) -> bool:
        try:
            await self.client.show(self.model)
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                return False  # server up, model not pulled
            raise
        return True
