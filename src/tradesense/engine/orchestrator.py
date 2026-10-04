"""Parallel analysis: every selected (timeframe, strategy) pair runs concurrently.

``run_analysis(instrument, selections)``

1. resolves the instrument and validates timeframes / strategy names (problems
   become error entries for that slot only);
2. fetches candles for all needed timeframes concurrently (a failing timeframe
   only affects its own column);
3. computes the shared feature pack lazily, once per timeframe;
4. runs every pair with ``asyncio.gather``: strategy features -> LLM decision;
5. streams each result as soon as it finishes (``on_event``), then a confidence-
   weighted consensus per timeframe, then the final :class:`AnalysisResult`.
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from tradesense.config import AnalysisConfig, TimeframeConfig
from tradesense.data.models import completed
from tradesense.engine.chart_store import ChartStore
from tradesense.llm.decision import DecisionEngine
from tradesense.llm.schemas import Decision, KeyLevels, Probability, Signal
from tradesense.llm.usage import RunUsage
from tradesense.strategies.base import (
    StrategyCategory,
    StrategyContext,
    StrategySource,
    UnknownStrategyError,
)
from tradesense.strategies.feature_pack import compute_feature_pack
from tradesense.strategies.registry import StrategyRegistry

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
ResultStatus = Literal["ok", "insufficient_data", "error"]


class CandleSource(Protocol):
    async def resolve_instrument(self, raw: str) -> str: ...

    async def get_candles(
        self, instrument: str, timeframes: list[str], count: int | None = None
    ) -> dict[str, pd.DataFrame]: ...


# --------------------------------------------------------------------------- results


class StrategyResult(BaseModel):
    timeframe: str
    strategy: str
    display_name: str
    source: StrategySource | None = None
    category: StrategyCategory | None = None
    status: ResultStatus
    decision: Decision | None = None
    message: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    pre_signal: Signal | None = None
    events: list[str] = Field(default_factory=list)
    candles_available: int | None = None
    candles_required: int | None = None
    candle_time: datetime | None = None  # open time of the last completed candle analysed
    reference_image_url: str | None = None
    live_chart_url: str | None = None
    started_at: datetime
    finished_at: datetime
    duration_ms: int


class TimeframeResult(BaseModel):
    timeframe: str
    consensus: Decision | None = None
    strategies: list[StrategyResult] = Field(default_factory=list)
    candles: int = 0
    last_candle_time: datetime | None = None
    error: str | None = None


class AnalysisResult(BaseModel):
    run_id: str
    instrument: str
    generated_at: datetime
    duration_ms: int
    usage: RunUsage
    timeframes: dict[str, TimeframeResult]


# --------------------------------------------------------------------------- consensus


def compute_consensus(named: Sequence[tuple[str, Decision]]) -> Decision | None:
    """Confidence-weighted average of strategy probabilities.

    ``p = sum(conf_i * p_i) / sum(conf_i)``; signal = highest probability.
    Consensus confidence = mean confidence x agreement, where agreement is the share of
    confidence weight whose own signal matches the consensus signal. Decisions without
    a valid LLM answer (fallbacks) are excluded.
    """
    valid = [(name, d) for name, d in named if d.decided_by != "none"]
    skipped = len(named) - len(valid)
    if not valid:
        return None
    total = sum(d.confidence for _, d in valid)
    counts = Counter(d.signal for _, d in valid)
    tally = (
        f"{counts['BUY']} BUY / {counts['SELL']} SELL / {counts['SIDEWAYS']} SIDEWAYS "
        f"across {len(valid)} strateg{'y' if len(valid) == 1 else 'ies'}"
    )
    extra = [f"{skipped} strategy decision(s) unavailable"] if skipped else []
    if total <= 0:
        return Decision(
            signal="SIDEWAYS",
            probability=Probability(buy=0.0, sell=0.0, sideways=1.0),
            confidence=0.0,
            observations=[tally, "Every decision had zero confidence", *extra],
            reasoning="No strategy expressed confidence, so the consensus is SIDEWAYS.",
            strategy_source="consensus",
            decided_by="consensus",
            model="confidence-weighted average",
        )
    probability = Probability.normalized(
        sum(d.confidence * d.probability.buy for _, d in valid) / total,
        sum(d.confidence * d.probability.sell for _, d in valid) / total,
        sum(d.confidence * d.probability.sideways for _, d in valid) / total,
    )
    signal = probability.top()
    agreement = sum(d.confidence for _, d in valid if d.signal == signal) / total
    mean_confidence = total / len(valid)
    strongest_name, strongest = max(valid, key=lambda item: item[1].confidence)
    supports = [d.key_levels.support for _, d in valid if d.key_levels.support is not None]
    resistances = [d.key_levels.resistance for _, d in valid if d.key_levels.resistance is not None]
    return Decision(
        signal=signal,
        probability=probability,
        confidence=round(mean_confidence * agreement, 4),
        observations=[
            tally,
            f"{agreement:.0%} of confidence weight agrees with {signal}",
            f"Most confident: {strongest_name} ({strongest.signal}, {strongest.confidence:.0%})",
            *extra,
        ],
        reasoning=(
            f"Confidence-weighted average of {len(valid)} strategy decisions. "
            f"Mean confidence {mean_confidence:.0%}, agreement {agreement:.0%}."
        ),
        key_levels=KeyLevels(
            support=statistics.median(supports) if supports else None,
            resistance=statistics.median(resistances) if resistances else None,
        ),
        strategy_source="consensus",
        decided_by="consensus",
        model="confidence-weighted average",
    )


# --------------------------------------------------------------------------- orchestrator


class _LazyFeaturePack:
    """Computes the standard feature pack at most once per timeframe (thread-safe)."""

    def __init__(self, df: pd.DataFrame, history_length: int) -> None:
        self._df = df
        self._history_length = history_length
        self._lock = threading.Lock()
        self._value: dict[str, Any] | None = None

    def __call__(self) -> dict[str, Any]:
        with self._lock:
            if self._value is None:
                self._value = compute_feature_pack(self._df, self._history_length)
            return self._value


async def _noop(_: dict[str, Any]) -> None:
    return None


class Orchestrator:
    def __init__(
        self,
        *,
        source: CandleSource,
        registry: StrategyRegistry,
        decisions: DecisionEngine,
        timeframes: Mapping[str, TimeframeConfig],
        analysis: AnalysisConfig | None = None,
        charts: ChartStore | None = None,
    ) -> None:
        self.source = source
        self.registry = registry
        self.decisions = decisions
        self.timeframes = dict(timeframes)
        self.analysis = analysis or AnalysisConfig()
        # ``is None`` matters: an empty ChartStore is falsy (it defines __len__).
        self.charts = charts if charts is not None else ChartStore()

    async def run_analysis(
        self,
        instrument: str,
        selections: Mapping[str, Sequence[str]],
        *,
        on_event: EventCallback | None = None,
        run_id: str | None = None,
    ) -> AnalysisResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        emit = _safe_emitter(on_event or _noop)
        resolved = await self.source.resolve_instrument(instrument)

        plan: dict[str, list[str]] = {}
        timeframe_errors: dict[str, str] = {}
        for timeframe, names in selections.items():
            plan[timeframe] = list(dict.fromkeys(str(n) for n in names))
            if timeframe not in self.timeframes:
                timeframe_errors[timeframe] = (
                    f"Unknown timeframe '{timeframe}'. Configured: {', '.join(self.timeframes)}"
                )
        await emit(
            {
                "type": "analysis_started",
                "run_id": run_id,
                "instrument": resolved,
                "selections": plan,
            }
        )

        needed = [tf for tf, names in plan.items() if names and tf not in timeframe_errors]
        fetched = await asyncio.gather(
            *(self.source.get_candles(resolved, [tf]) for tf in needed), return_exceptions=True
        )
        frames: dict[str, pd.DataFrame] = {}
        for timeframe, outcome in zip(needed, fetched, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("Candles for {} {} failed: {}", resolved, timeframe, outcome)
                timeframe_errors[timeframe] = f"Could not load candles: {outcome}"
            else:
                frames[timeframe] = outcome[timeframe]

        results = await asyncio.gather(
            *(
                self._run_timeframe(
                    run_id, resolved, tf, names, frames.get(tf), timeframe_errors.get(tf), emit
                )
                for tf, names in plan.items()
            )
        )
        by_timeframe = {r.timeframe: r for r in results}
        decisions = [s.decision for r in results for s in r.strategies if s.decision is not None]
        analysis = AnalysisResult(
            run_id=run_id,
            instrument=resolved,
            generated_at=datetime.now(UTC),
            duration_ms=int((time.perf_counter() - started) * 1000),
            usage=RunUsage.from_decisions(decisions),
            timeframes=by_timeframe,
        )
        await emit(
            {
                "type": "analysis_complete",
                "run_id": run_id,
                "result": analysis.model_dump(mode="json"),
            }
        )
        logger.info(
            "Analysis {} for {} finished in {} ms ({} decisions, {} tokens)",
            run_id,
            resolved,
            analysis.duration_ms,
            len(decisions),
            analysis.usage.total_tokens,
        )
        return analysis

    async def _run_timeframe(
        self,
        run_id: str,
        instrument: str,
        timeframe: str,
        names: list[str],
        frame: pd.DataFrame | None,
        error: str | None,
        emit: EventCallback,
    ) -> TimeframeResult:
        if error is not None or frame is None:
            message = error or "No candles available"
            now = datetime.now(UTC)
            failed = []
            for name in names:
                result = StrategyResult(
                    timeframe=timeframe,
                    strategy=name,
                    display_name=name,
                    status="error",
                    message=message,
                    started_at=now,
                    finished_at=now,
                    duration_ms=0,
                )
                failed.append(result)
                await emit(_result_event(run_id, result))
            return TimeframeResult(timeframe=timeframe, strategies=failed, error=message)

        candles = completed(frame)
        context = StrategyContext(
            instrument=instrument,
            history_length=self.analysis.history_length,
            prompt_candles=self.analysis.prompt_candles,
            document_prompt_candles=self.analysis.document_prompt_candles,
            chart_candles=self.analysis.chart_candles,
            feature_pack=_LazyFeaturePack(candles, self.analysis.history_length),
        )
        strategy_results = list(
            await asyncio.gather(
                *(
                    self._run_one(run_id, instrument, timeframe, name, candles, context, emit)
                    for name in names
                )
            )
        )
        consensus = compute_consensus(
            [(r.display_name, r.decision) for r in strategy_results if r.decision is not None]
        )
        await emit(
            {
                "type": "consensus",
                "run_id": run_id,
                "timeframe": timeframe,
                "consensus": consensus.model_dump(mode="json") if consensus else None,
            }
        )
        return TimeframeResult(
            timeframe=timeframe,
            consensus=consensus,
            strategies=strategy_results,
            candles=len(candles),
            last_candle_time=candles.index[-1].to_pydatetime() if len(candles) else None,
        )

    async def _run_one(
        self,
        run_id: str,
        instrument: str,
        timeframe: str,
        name: str,
        candles: pd.DataFrame,
        context: StrategyContext,
        emit: EventCallback,
    ) -> StrategyResult:
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        fields: dict[str, Any] = {"display_name": name}
        try:
            strategy = self.registry.get(name)
            fields.update(
                display_name=strategy.display_name,
                source=strategy.source,
                category=strategy.category,
            )
            if strategy.source == "image":
                fields["reference_image_url"] = f"/api/library/{name}/file"
            available, reason = self.registry.availability(strategy)
            if not available:
                raise RuntimeError(reason or "strategy unavailable")
            params = self.registry.params_for(name, timeframe)
            fields["params"] = params
            features = await strategy.compute(candles, params, timeframe, context=context)
            fields.update(
                candles_available=features.candles_available,
                candles_required=features.candles_required,
                candle_time=features.last_candle_time,
                pre_signal=features.pre_signal,
                events=features.events,
            )
            if features.status == "insufficient_data":
                fields.update(status="insufficient_data", message=features.message)
            else:
                live = next((i for i in features.images if i.label == "live_chart"), None)
                if live is not None:
                    fields["live_chart_url"] = f"/api/charts/{self.charts.put(live.to_bytes())}.png"
                fields["decision"] = await self.decisions.decide(features, instrument, candles)
                fields["status"] = "ok"
        except UnknownStrategyError as exc:
            fields.update(status="error", message=str(exc))
        except Exception as exc:  # one failing strategy never breaks the others
            logger.opt(exception=exc).warning("Strategy {} on {} failed", name, timeframe)
            fields.update(status="error", message=f"{type(exc).__name__}: {exc}")
        result = StrategyResult(
            timeframe=timeframe,
            strategy=name,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            duration_ms=int((time.perf_counter() - started) * 1000),
            **fields,
        )
        await emit(_result_event(run_id, result))
        return result


def _result_event(run_id: str, result: StrategyResult) -> dict[str, Any]:
    return {
        "type": "strategy_result",
        "run_id": run_id,
        "timeframe": result.timeframe,
        "result": result.model_dump(mode="json"),
    }


def _safe_emitter(callback: EventCallback) -> EventCallback:
    """Event delivery problems (e.g. a closed WebSocket) must not break the analysis."""

    async def emit(event: dict[str, Any]) -> None:
        try:
            await callback(event)
        except Exception as exc:
            logger.debug("Dropping {} event: {}", event.get("type"), exc)

    return emit
