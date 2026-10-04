"""Test doubles for the LLM layer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from tradesense.config import LLMConfig, LLMSettings, PricingConfig, ProviderConfig, RoleConfig
from tradesense.llm.base import LLMProvider
from tradesense.llm.messages import LLMRequest
from tradesense.llm.router import LLMRouter

VALID_DECISION: dict[str, Any] = {
    "signal": "BUY",
    "probability": {"buy": 0.62, "sell": 0.13, "sideways": 0.25},
    "confidence": 0.71,
    "observations": ["EMA9 above EMA21", "Price above both EMAs"],
    "reasoning": "Trend structure is bullish and momentum confirms it.",
    "key_levels": {"support": 1.0841, "resistance": 1.0875},
    "pattern_match": None,
}
VALID_JSON = json.dumps(VALID_DECISION)

FAST_SETTINGS = LLMSettings(retries=2, retry_backoff_seconds=0, timeout_seconds=5)


class ScriptedProvider(LLMProvider):
    """Returns scripted outputs in order (the last one repeats). Exceptions are raised."""

    type_name = "scripted"
    requires_api_key = False

    def __init__(
        self,
        name: str,
        script: Sequence[str | Exception],
        *,
        capabilities: Sequence[str] = ("text",),
        settings: LLMSettings = FAST_SETTINGS,
        max_concurrency: int | None = None,
        delay: float = 0.0,
        healthy: bool = True,
    ) -> None:
        config = ProviderConfig(
            type="scripted",
            model=f"{name}-model",
            capabilities=list(capabilities),  # type: ignore[arg-type]
            max_concurrency=max_concurrency,
            pricing=PricingConfig(input_per_mtok=1.0, output_per_mtok=2.0),
        )
        super().__init__(name, config, settings, env={})
        self.script = list(script)
        self.calls: list[LLMRequest] = []
        self.delay = delay
        self.healthy = healthy
        self.in_flight = 0
        self.peak = 0

    async def _generate(self, request: LLMRequest) -> tuple[str, int, int]:
        self.calls.append(request)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        finally:
            self.in_flight -= 1
        if isinstance(item, Exception):
            raise item
        return item, 100, 50

    async def health_check(self) -> bool:
        return self.healthy


def make_router(
    providers: Sequence[ScriptedProvider],
    roles: dict[str, list[str]] | None = None,
    settings: LLMSettings = FAST_SETTINGS,
) -> LLMRouter:
    names = [p.name for p in providers]
    roles = roles or {"text": names}
    config = LLMConfig(
        roles={  # type: ignore[arg-type]
            role: RoleConfig(primary=chain[0], fallbacks=chain[1:]) for role, chain in roles.items()
        },
        settings=settings,
        providers={p.name: p.config for p in providers},
    )

    async def no_sleep(_: float) -> None:
        return None

    return LLMRouter(config, {p.name: p for p in providers}, sleep=no_sleep)
