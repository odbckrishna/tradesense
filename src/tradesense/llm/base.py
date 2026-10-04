"""Abstract LLM provider, capabilities and the error taxonomy the router acts on.

Error classes drive routing:

* :class:`LLMTransientError` (timeouts, 429, 5xx, connection) -> retry with backoff,
  then fall back to the next provider
* :class:`LLMAuthError` (401/403) and :class:`LLMQuotaError` (credits exhausted) -> skip
  straight to the next provider, no retries, and skip that provider until reload
* any other :class:`LLMError` (bad request, empty response, misconfiguration) -> fall back
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Literal

from tradesense.llm.messages import LLMRequest, LLMResponse
from tradesense.llm.usage import compute_cost

if TYPE_CHECKING:
    from tradesense.config import LLMSettings, ProviderConfig

Capability = Literal["text", "vision", "decision"]


class LLMError(Exception):
    """Base class for provider failures."""

    def __init__(
        self, message: str, *, provider: str | None = None, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMTransientError(LLMError):
    """Worth retrying: timeouts, connection failures, 5xx."""


class LLMRateLimitError(LLMTransientError):
    """429 / quota exhausted."""


class LLMAuthError(LLMError):
    """401 / 403: the key was rejected. Never retried."""


class LLMQuotaError(LLMError):
    """Credits / quota exhausted (often reported as 429). Never retried: retrying cannot
    succeed until the account is topped up."""


class LLMBadRequestError(LLMError):
    """The provider rejected the request (400/404/413/422)."""


class LLMResponseError(LLMError):
    """The provider answered but the answer is unusable (empty, truncated)."""


class LLMUnavailableError(LLMError):
    """The provider is not configured (e.g. its API key env var is not set)."""


class LLMConfigError(ValueError):
    """llm.yml is structurally valid YAML but cannot be turned into providers."""


_QUOTA_EXHAUSTED = re.compile(
    r"insufficient_quota|insufficient credits|credit_balance|no credits|credit balance|billing"
    r"|quota exceeded",
    re.IGNORECASE,
)


def classify_status(status: int | None, message: str, provider: str | None = None) -> LLMError:
    """Map an HTTP status code to the error class the router understands."""
    if status in (401, 403):
        return LLMAuthError(message, provider=provider, status_code=status)
    if status in (402, 429) and _QUOTA_EXHAUSTED.search(message):
        return LLMQuotaError(message, provider=provider, status_code=status)
    if status == 429:
        return LLMRateLimitError(message, provider=provider, status_code=status)
    if status is not None and (status >= 500 or status in (408, 409)):
        return LLMTransientError(message, provider=provider, status_code=status)
    if status in (400, 404, 413, 422):
        return LLMBadRequestError(message, provider=provider, status_code=status)
    return LLMError(message, provider=provider, status_code=status)


class LLMProvider(ABC):
    """One configured model endpoint.

    Subclasses translate :class:`LLMRequest` to their SDK in :meth:`_generate` and map
    SDK exceptions onto :class:`LLMError` subclasses. Clients are created lazily so
    building providers never touches the network.
    """

    #: Set by :func:`tradesense.llm.factory.register_provider`.
    type_name: ClassVar[str] = ""
    #: Whether a missing ``api_key_env`` value makes the provider unusable.
    requires_api_key: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        settings: LLMSettings,
        env: Mapping[str, str],
    ) -> None:
        self.name = name
        self.config = config
        self.settings = settings
        self.model = config.model_name
        self.capabilities: set[Capability] = set(config.capabilities)
        self.api_key: str | None = None
        if config.api_key_env:
            value = env.get(config.api_key_env, "").strip()
            self.api_key = value or None
        self.timeout = float(config.timeout_seconds or settings.timeout_seconds)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} model={self.model}>"

    # ------------------------------------------------------------------ configuration

    @property
    def configured(self) -> bool:
        return self.missing_config_reason() is None

    def missing_config_reason(self) -> str | None:
        if self.requires_api_key and self.config.api_key_env and not self.api_key:
            return f"environment variable {self.config.api_key_env} is not set"
        if self.requires_api_key and not self.config.api_key_env:
            return "no api_key_env configured"
        return None

    @property
    def is_local(self) -> bool:
        """Runs on this machine / LAN (shown as "Local" in the model picker)."""
        return False

    @property
    def generates_text(self) -> bool:
        return bool(self.capabilities & {"text", "vision"})

    def supports(self, request: LLMRequest) -> bool:
        if request.remote_only and self.is_local:
            return False
        if request.has_images:
            return "vision" in self.capabilities
        if self.generates_text:
            return True
        return request.task == "decision" and "decision" in self.capabilities

    def json_mode_enabled(self, request: LLMRequest) -> bool:
        if not request.json_mode:
            return False
        if self.config.json_mode is not None:
            return self.config.json_mode
        return self.settings.force_json

    def temperature_for(self, request: LLMRequest) -> float:
        return request.temperature if request.temperature is not None else self.settings.temperature

    def max_tokens_for(self, request: LLMRequest) -> int:
        return request.max_tokens or self.settings.max_output_tokens

    def option(self, key: str, default: object = None) -> object:
        return self.config.options.get(key, default)

    # ------------------------------------------------------------------ calls

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one request (no retries: the router owns retry policy)."""
        reason = self.missing_config_reason()
        if reason:
            raise LLMUnavailableError(f"{self.name}: {reason}", provider=self.name)
        started = time.perf_counter()
        text, input_tokens, output_tokens = await self._generate(request)
        latency_ms = int((time.perf_counter() - started) * 1000)
        cost = (
            compute_cost(self.config.pricing, input_tokens, output_tokens)
            if self.settings.track_costs
            else None
        )
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    @abstractmethod
    async def _generate(self, request: LLMRequest) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)`` or raise :class:`LLMError`."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Cheap reachability/auth probe that does not consume tokens."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release client resources."""
