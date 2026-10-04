"""Turns strategy features into a validated :class:`Decision` via the LLM router."""

from __future__ import annotations

import time

import pandas as pd
from loguru import logger

from tradesense.config import RoleName
from tradesense.llm.base import LLMError
from tradesense.llm.messages import LLMResponse
from tradesense.llm.prompts import build_decision_request
from tradesense.llm.router import AllProvidersFailedError, LLMRouter, NoProviderError
from tradesense.llm.schemas import Decision, parse_decision_payload
from tradesense.strategies.base import StrategyFeatures


class DecisionEngine:
    """Strategies compute FEATURES; this asks the LLM for the DECISION."""

    def __init__(
        self,
        router: LLMRouter,
        *,
        prompt_candles: int = 20,
        document_prompt_candles: int = 50,
    ) -> None:
        self.router = router
        self.prompt_candles = prompt_candles
        self.document_prompt_candles = document_prompt_candles

    async def decide(
        self,
        features: StrategyFeatures,
        instrument: str,
        candles: pd.DataFrame,
        *,
        remote_only: bool = False,
    ) -> Decision:
        """Never raises for LLM problems: falls back to SIDEWAYS with confidence 0.
        ``remote_only`` skips local providers (used for bulk backtest checks)."""
        role: RoleName = "vision" if features.images else "text"
        request = build_decision_request(
            features,
            instrument,
            candles,
            prompt_candles=self.prompt_candles,
            document_prompt_candles=self.document_prompt_candles,
        ).model_copy(update={"remote_only": remote_only})
        started = time.perf_counter()
        try:
            result = await self.router.generate_structured(role, request, parse_decision_payload)
        except AllProvidersFailedError as exc:
            responses = exc.responses
            logger.warning(
                "Decision failed for {} {}: {}", features.timeframe, features.strategy, exc
            )
            return Decision.fallback(
                features.source,
                str(exc),
                observation="LLM parse error" if exc.parse_failed else "LLM unavailable",
                latency_ms=_elapsed(started),
                input_tokens=sum(r.input_tokens for r in responses),
                output_tokens=sum(r.output_tokens for r in responses),
                cost_usd=_cost(responses),
                llm_calls=len(responses),
            )
        except (NoProviderError, LLMError) as exc:
            return Decision.fallback(
                features.source,
                str(exc),
                observation="LLM unavailable",
                latency_ms=_elapsed(started),
            )

        payload = result.value
        responses = result.responses
        return Decision(
            signal=payload.signal,
            probability=payload.probability,
            confidence=payload.confidence,
            observations=payload.observations,
            reasoning=payload.reasoning,
            key_levels=payload.key_levels,
            pattern_match=payload.pattern_match if features.source == "image" else None,
            strategy_source=features.source,
            decided_by=result.provider,
            model=result.response.model,
            latency_ms=_elapsed(started),
            cost_usd=_cost(responses),
            input_tokens=sum(r.input_tokens for r in responses),
            output_tokens=sum(r.output_tokens for r in responses),
            llm_calls=len(responses),
            fallback_used=result.fallback_used,
        )


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _cost(responses: list[LLMResponse]) -> float | None:
    costs = [r.cost_usd for r in responses if r.cost_usd is not None]
    return round(sum(costs), 6) if costs else None
