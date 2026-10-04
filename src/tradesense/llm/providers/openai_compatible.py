"""Any OpenAI-compatible endpoint: vLLM, Groq, OpenRouter, Together, LM Studio, ...

Same translation as :mod:`openai_provider` against a custom ``base_url``. The API key
is optional for local servers (localhost / 127.0.0.1); remote endpoints need it.
"""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import urlparse

import openai

from tradesense.llm.factory import register_provider
from tradesense.llm.providers.openai_provider import OpenAIProvider

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


@register_provider("openai_compatible")
class OpenAICompatibleProvider(OpenAIProvider):
    default_token_param: ClassVar[str] = "max_tokens"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.config.base_url:
            raise ValueError("openai_compatible providers require 'base_url'")

    @property
    def is_local(self) -> bool:
        host = urlparse(self.config.base_url or "").hostname or ""
        return host in _LOCAL_HOSTS

    def missing_config_reason(self) -> str | None:
        if self.is_local:
            return None
        if self.config.api_key_env and not self.api_key:
            return f"environment variable {self.config.api_key_env} is not set"
        return None

    def make_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.config.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
