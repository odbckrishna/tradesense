"""Role -> provider routing with retries, fallbacks and concurrency limits.

``generate(role, request)`` tries the role's primary provider first, retrying
transient errors (timeouts, 429, 5xx) with exponential backoff, then each fallback
in order. Auth errors (401/403) skip straight to the next fallback. Requests with
images skip providers without the ``vision`` capability. A global semaphore and
optional per-provider semaphores cap parallel calls; neither is held while backing off.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from tradesense.config import LLMConfig, RoleName
from tradesense.llm.base import (
    LLMAuthError,
    LLMError,
    LLMProvider,
    LLMQuotaError,
    LLMTransientError,
)
from tradesense.llm.messages import LLMRequest, LLMResponse
from tradesense.llm.schemas import DecisionParseError
from tradesense.llm.usage import UsageTracker

Sleep = Callable[[float], Awaitable[None]]

CORRECTION_PROMPT = (
    "Your previous reply could not be used: {error}. Reply again with ONLY the JSON object "
    "that matches the required schema - no prose, no markdown, no code fences."
)


class NoProviderError(LLMError):
    """The role has no provider able to serve this request."""


class AllProvidersFailedError(LLMError):
    """Every provider in the chain failed."""

    def __init__(
        self,
        role: str,
        errors: list[str],
        *,
        parse_failed: bool = False,
        responses: list[LLMResponse] | None = None,
    ) -> None:
        summary = "; ".join(errors) if errors else "no providers available"
        super().__init__(f"all providers failed for role '{role}': {summary}")
        self.role = role
        self.errors = errors
        self.parse_failed = parse_failed
        self.responses = responses or []


@dataclass
class ProviderState:
    healthy: bool | None = None
    checked_at: datetime | None = None
    error: str | None = None
    disabled: bool = False
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None


@dataclass
class StructuredResult[T]:
    value: T
    response: LLMResponse
    provider: str
    fallback_used: bool
    responses: list[LLMResponse] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class LLMRouter:
    def __init__(
        self,
        config: LLMConfig,
        providers: Mapping[str, LLMProvider],
        usage: UsageTracker | None = None,
        *,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.config = config
        self.settings = config.settings
        self.providers = dict(providers)
        self.usage = usage or UsageTracker()
        self._sleep = sleep
        self._global = asyncio.Semaphore(self.settings.max_concurrency)
        self._per_provider = {
            name: asyncio.Semaphore(p.config.max_concurrency)
            for name, p in self.providers.items()
            if p.config.max_concurrency
        }
        self.state: dict[str, ProviderState] = {name: ProviderState() for name in self.providers}
        self._last_used: dict[str, str] = {}

    # ------------------------------------------------------------------ chains

    def chain(self, role: RoleName, request: LLMRequest | None = None) -> list[LLMProvider]:
        """Providers to try for ``role``, in order, filtered for this request."""
        chain = []
        for name in self.config.role_chain(role):
            provider = self.providers[name]
            if not provider.configured or self.state[name].disabled:
                continue
            if request is not None and not provider.supports(request):
                continue
            if role == "vision" and "vision" not in provider.capabilities:
                continue
            if role == "extraction" and not provider.generates_text:
                continue
            chain.append(provider)
        return chain

    def vision_available(self) -> bool:
        """True if at least one configured vision provider is not known to be down."""
        if "vision" not in self.config.roles:
            return False
        return any(self.state[p.name].healthy is not False for p in self.chain("vision"))

    # ------------------------------------------------------------------ calls

    async def generate(self, role: RoleName, request: LLMRequest) -> LLMResponse:
        """Primary first, retries on transient errors, then fallbacks in order."""
        chain = self._require_chain(role, request)
        errors: list[str] = []
        for provider in chain:
            try:
                response = await self._call_with_retries(provider, request)
            except LLMError as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            self._last_used[role] = provider.name
            if provider is not chain[0]:
                logger.warning("Role '{}' answered by fallback provider '{}'", role, provider.name)
            return response
        raise AllProvidersFailedError(role, errors)

    async def generate_structured[T](
        self,
        role: RoleName,
        request: LLMRequest,
        parser: Callable[[str], T],
    ) -> StructuredResult[T]:
        """Like :meth:`generate`, but the answer must parse. Invalid output gets one
        error-correction retry on the same provider, then the next fallback is tried."""
        chain = self._require_chain(role, request)
        errors: list[str] = []
        responses: list[LLMResponse] = []
        parse_failed = False
        for provider in chain:
            attempt_request = request
            for correction in (False, True):
                try:
                    response = await self._call_with_retries(provider, attempt_request)
                except LLMError as exc:
                    errors.append(f"{provider.name}: {exc}")
                    break
                responses.append(response)
                try:
                    value = parser(response.text)
                except (DecisionParseError, ValueError) as exc:
                    parse_failed = True
                    errors.append(f"{provider.name}: invalid output ({exc})")
                    logger.warning(
                        "{} returned invalid output for role '{}' ({}){}",
                        provider.name,
                        role,
                        exc,
                        "" if correction else "; asking it to correct",
                    )
                    if correction:
                        break
                    attempt_request = request.with_followup(
                        response.text[:4000], CORRECTION_PROMPT.format(error=exc)
                    )
                    continue
                self._last_used[role] = provider.name
                return StructuredResult(
                    value=value,
                    response=response,
                    provider=provider.name,
                    fallback_used=provider is not chain[0],
                    responses=responses,
                    errors=errors,
                )
        raise AllProvidersFailedError(role, errors, parse_failed=parse_failed, responses=responses)

    def _require_chain(self, role: RoleName, request: LLMRequest) -> list[LLMProvider]:
        chain = self.chain(role, request)
        if not chain:
            needs = "vision-capable " if request.has_images else ""
            raise NoProviderError(f"no usable {needs}provider configured for role '{role}'")
        return chain

    async def _call_with_retries(self, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        attempts = 1 + self.settings.retries
        for attempt in range(1, attempts + 1):
            try:
                response = await self._call_once(provider, request)
            except (LLMAuthError, LLMQuotaError) as exc:
                self._mark_failure(provider, str(exc), disable=True)
                problem = (
                    "rejected credentials" if isinstance(exc, LLMAuthError) else "is out of quota"
                )
                logger.error("{} {}; skipping it until LLM config reload", provider.name, problem)
                raise
            except LLMTransientError as exc:
                self._mark_failure(provider, str(exc))
                if attempt >= attempts:
                    raise
                delay = self.settings.retry_backoff_seconds * (2 ** (attempt - 1))
                delay *= random.uniform(0.8, 1.2)
                logger.warning(
                    "{} transient error ({}); retry {}/{} in {:.1f}s",
                    provider.name,
                    exc,
                    attempt,
                    attempts - 1,
                    delay,
                )
                await self._sleep(delay)
                continue
            except LLMError as exc:
                self._mark_failure(provider, str(exc))
                raise
            self._mark_success(provider, response)
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    async def _call_once(self, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        semaphore = self._per_provider.get(provider.name)
        try:
            if semaphore is not None:
                async with semaphore, self._global:
                    return await self._timed(provider, request)
            async with self._global:
                return await self._timed(provider, request)
        except LLMError:
            raise
        except Exception as exc:  # adapter bug or unexpected SDK error: never crash a run
            logger.exception("Unexpected error from {}", provider.name)
            raise LLMError(f"{provider.name}: {type(exc).__name__}: {exc}") from exc

    async def _timed(self, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        try:
            async with asyncio.timeout(provider.timeout):
                return await provider.generate(request)
        except TimeoutError as exc:
            raise LLMTransientError(
                f"{provider.name}: no response within {provider.timeout:.0f}s",
                provider=provider.name,
            ) from exc

    def _mark_success(self, provider: LLMProvider, response: LLMResponse) -> None:
        state = self.state[provider.name]
        state.healthy = True
        state.error = None
        state.last_success_at = datetime.now(UTC)
        self.usage.record(response)

    def _mark_failure(self, provider: LLMProvider, error: str, *, disable: bool = False) -> None:
        state = self.state[provider.name]
        state.error = error[:300]
        state.last_error_at = datetime.now(UTC)
        if disable:
            state.disabled = True
            state.healthy = False
        self.usage.record_failure(provider.name)

    # ------------------------------------------------------------------ health

    async def check_health(self, names: list[str] | None = None) -> dict[str, bool | None]:
        """Run ``health_check`` concurrently for providers used by roles."""
        targets = names or self.config.used_providers()

        async def probe(name: str) -> None:
            provider = self.providers[name]
            state = self.state[name]
            state.checked_at = datetime.now(UTC)
            reason = provider.missing_config_reason()
            if reason:
                state.healthy, state.error = False, reason
                return
            try:
                async with asyncio.timeout(self.settings.health_check_timeout_seconds):
                    ok = await provider.health_check()
            except Exception as exc:
                state.healthy = False
                state.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                state.disabled = isinstance(exc, LLMAuthError) or _looks_like_auth(exc)
                return
            state.healthy = ok
            state.error = None if ok else "model not available"
            state.disabled = False

        await asyncio.gather(*(probe(n) for n in targets))
        for name in targets:
            state = self.state[name]
            if state.healthy:
                logger.info("LLM provider '{}' is healthy", name)
            else:
                logger.warning("LLM provider '{}' unavailable: {}", name, state.error)
        return {name: self.state[name].healthy for name in targets}

    # ------------------------------------------------------------------ status

    def role_status(self) -> dict[str, dict[str, Any]]:
        roles: dict[str, dict[str, Any]] = {}
        for role in self.config.roles:
            names = self.config.role_chain(role)
            usable = self.chain(role)
            healthy_usable = [p for p in usable if self.state[p.name].healthy is not False]
            last = self._last_used.get(role)
            active: str | None
            if last and any(p.name == last for p in usable):
                active = last
            elif healthy_usable:
                active = healthy_usable[0].name
            else:
                active = usable[0].name if usable else None
            active_provider = self.providers.get(active) if active else None
            roles[role] = {
                "primary": names[0],
                "chain": names,
                "active": active,
                "active_model": active_provider.model if active_provider else None,
                "fallback_in_use": bool(active and active != names[0]),
                "healthy": self.state[active].healthy if active else False,
                "available": bool(usable),
            }
        return roles

    def provider_status(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in self.config.used_providers():
            provider = self.providers[name]
            state = self.state[name]
            out[name] = {
                "type": provider.type_name,
                "model": provider.model,
                "capabilities": sorted(provider.capabilities),
                "configured": provider.configured,
                "config_issue": provider.missing_config_reason(),
                "healthy": state.healthy,
                "disabled": state.disabled,
                "error": state.error,
                "checked_at": state.checked_at.isoformat() if state.checked_at else None,
            }
        return out

    async def aclose(self) -> None:
        await asyncio.gather(*(p.aclose() for p in self.providers.values()), return_exceptions=True)


def _looks_like_auth(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status in (401, 403)
