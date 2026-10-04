"""OpenAI Chat Completions adapter (OpenAI and Azure OpenAI).

* system prompt is the first message
* images are ``image_url`` parts with base64 data URIs
* ``response_format={"type": "json_object"}`` when JSON is requested; if the endpoint
  rejects it, the adapter relies on prompt instructions from then on.

Azure: set ``azure: true`` plus ``endpoint``, ``deployment`` and ``api_version``.

Options: ``token_param`` (max_tokens | max_completion_tokens), ``send_temperature``
(bool, default true; set false for reasoning models), ``extra_body`` (dict passed
through to the API).
"""

from __future__ import annotations

from typing import Any, ClassVar

import openai
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


@register_provider("openai")
class OpenAIProvider(LLMProvider):
    default_token_param: ClassVar[str] = "max_completion_tokens"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: openai.AsyncOpenAI | None = None
        self._json_format_ok = True

    def make_client(self) -> openai.AsyncOpenAI:
        if self.config.azure:
            return openai.AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.config.endpoint or "",
                api_version=self.config.api_version,
                timeout=self.timeout,
                max_retries=0,
            )
        return openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.config.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = self.make_client()
        return self._client

    # ------------------------------------------------------------------ translation

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            if message.role == "assistant" or not message.images:
                messages.append({"role": message.role, "content": message.text})
                continue
            parts: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    parts.append({"type": "image_url", "image_url": {"url": part.data_uri}})
            messages.append({"role": "user", "content": parts})

        params: dict[str, Any] = {"model": self.model, "messages": messages}
        token_param = str(self.option("token_param", self.default_token_param))
        params[token_param] = self.max_tokens_for(request)
        if self.option("send_temperature", True):
            params["temperature"] = self.temperature_for(request)
        if self.json_mode_enabled(request) and self._json_format_ok:
            params["response_format"] = {"type": "json_object"}
        extra_body = self.option("extra_body")
        if isinstance(extra_body, dict) and extra_body:
            params["extra_body"] = extra_body
        return params

    # ------------------------------------------------------------------ calls

    async def _generate(self, request: LLMRequest) -> tuple[str, int, int]:
        params = self.build_params(request)
        try:
            response = await self.client.chat.completions.create(**params)
        except openai.BadRequestError as exc:
            if "response_format" in params and "response_format" in str(exc).lower():
                logger.warning(
                    "{}: response_format rejected; relying on prompt instructions", self.name
                )
                self._json_format_ok = False
                return await self._generate(request)
            raise self._map(exc) from exc
        except openai.APIError as exc:
            raise self._map(exc) from exc

        if not response.choices:
            raise LLMResponseError(f"{self.name}: no choices returned", provider=self.name)
        text = response.choices[0].message.content or ""
        if not text.strip():
            reason = response.choices[0].finish_reason
            raise LLMResponseError(
                f"{self.name}: empty response (finish_reason={reason})", provider=self.name
            )
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return text, input_tokens, output_tokens

    def _map(self, exc: Exception) -> LLMError:
        if isinstance(exc, openai.APITimeoutError):
            return LLMTransientError(f"{self.name}: request timed out", provider=self.name)
        if isinstance(exc, openai.APIConnectionError):
            return LLMTransientError(f"{self.name}: connection error", provider=self.name)
        if isinstance(exc, openai.APIStatusError):
            message = str(exc).replace("\n", " ")[:300]
            return classify_status(exc.status_code, f"{self.name}: {message}", self.name)
        return LLMError(f"{self.name}: {exc}", provider=self.name)

    async def health_check(self) -> bool:
        await self.client.models.list()
        return True

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
