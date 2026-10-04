"""Backtest: which strategies called BUY / SELL right on this instrument lately?

For every configured timeframe the last ``backtest.days`` of candles are replayed:

1. **Pool strategies.** Built-ins are replayed candle by candle on the live-sized window
   in a process pool; YAML rules are vectorised. Document / image strategies have no
   mechanical signal and are skipped.
2. **Scoring.** Directional hit rate (see :mod:`tradesense.backtest.scoring`), ranked by
   its Wilson lower bound.
3. **New strategies.** Rule templates are searched on the first half of the period and
   judged on the second; optionally a text LLM proposes rules judged the same way.
4. **LLM check.** The live LLM decision is replayed on a sample of the top strategies'
   own BUY / SELL candles, to see whether the LLM improves on the raw rule.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from tradesense.backtest.designer import design_rules
from tradesense.backtest.models import (
    BacktestResult,
    HitStats,
    LLMCheck,
    SkippedStrategy,
    StrategyOrigin,
    StrategyScore,
    TimeframeBacktest,
)
from tradesense.backtest.scoring import (
    Threshold,
    forward_moves,
    period_masks,
    score_signals,
    threshold_at,
)
from tradesense.backtest.signals import SIGNAL_CODES, replay_builtin, rule_signals
from tradesense.backtest.templates import template_candidates
from tradesense.backtest.trace import StrategyTrace, build_trace
from tradesense.config import AppConfig
from tradesense.data.models import InstrumentInfo, PriceTick, completed
from tradesense.indicators.technical import atr, ema
from tradesense.llm.base import LLMError
from tradesense.llm.decision import DecisionEngine
from tradesense.llm.messages import LLMRequest, Message
from tradesense.llm.router import LLMRouter
from tradesense.strategies.base import IndicatorStrategy, Strategy, StrategyContext
from tradesense.strategies.registry import StrategyRegistry
from tradesense.strategies.rule_engine import RuleStrategy, load_rule

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
ExecutorFactory = Callable[[], Executor]

TEMPLATE_FINALISTS = 10  # best-on-train templates re-scored on validation
TEMPLATES_SHOWN = 5


class BacktestBusyError(RuntimeError):
    """Another backtest is running (HTTP 409)."""


class BacktestNotFoundError(LookupError):
    """The backtest run or candidate is no longer available (HTTP 404)."""


class BacktestSource(Protocol):
    async def resolve_instrument(self, raw: str) -> str: ...

    async def get_candles(
        self, instrument: str, timeframes: list[str], count: int | None = None
    ) -> dict[str, pd.DataFrame]: ...

    async def get_price(self, instrument: str) -> PriceTick: ...

    async def get_spreads(
        self, instrument: str, timeframe: str, count: int | None = None
    ) -> pd.Series: ...

    async def list_instruments(self) -> list[InstrumentInfo]: ...


@dataclass
class _Frame:
    """One timeframe's candles and the masks of its scored period."""

    timeframe: str
    candles: pd.DataFrame
    moves: np.ndarray
    whole: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    start: int
    end: int
    horizon: int
    spreads: np.ndarray | None = None  # bid/ask spread per candle, when OANDA has it


async def _noop(_: dict[str, Any]) -> None:
    return None


async def _quiet(stage: str, done: int, total: int, message: str) -> None:
    return None


def _worker_count(max_workers: int) -> int:
    return max_workers or max(1, (os.cpu_count() or 2) - 1)


def _default_executor(max_workers: int) -> ExecutorFactory:
    workers = _worker_count(max_workers)
    return lambda: ProcessPoolExecutor(max_workers=workers)


class Backtester:
    def __init__(
        self,
        *,
        source: BacktestSource,
        registry: StrategyRegistry,
        decisions: DecisionEngine,
        router: Callable[[], LLMRouter],
        app: AppConfig,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.source = source
        self.registry = registry
        self.decisions = decisions
        self.router = router
        self.app = app
        self.config = app.backtest
        self.window = app.oanda.candle_count
        self.workers = _worker_count(self.config.max_workers)
        self.executor_factory = executor_factory or _default_executor(self.config.max_workers)

    # ------------------------------------------------------------------ run

    async def run(
        self,
        instrument: str,
        *,
        llm_check: bool = True,
        llm_rules: bool = True,
        on_event: EventCallback | None = None,
        run_id: str | None = None,
    ) -> BacktestResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        callback = on_event or _noop

        async def emit(event: dict[str, Any]) -> None:
            try:
                await callback({"run_id": run_id, **event})
            except Exception:  # a dropped WebSocket must not stop the backtest
                logger.debug("Backtest event delivery failed", exc_info=True)

        async def progress(stage: str, done: int, total: int, message: str) -> None:
            await emit(
                {
                    "type": "backtest_progress",
                    "stage": stage,
                    "done": done,
                    "total": total,
                    "message": message,
                }
            )

        resolved = await self.source.resolve_instrument(instrument)
        await emit({"type": "backtest_started", "instrument": resolved})
        notes: list[str] = []

        await progress("candles", 0, 1, "Loading candles")
        frames, errors = await self._load(resolved)
        pip, pip_note = await self._pip_size(resolved, frames)
        thresholds, spread, spread_notes = await self._thresholds(resolved, frames)
        notes.extend(n for n in (pip_note, *spread_notes) if n)

        results = {
            tf: TimeframeBacktest(
                timeframe=tf, horizon_bars=self.config.horizon_for(tf), error=errors.get(tf)
            )
            for tf in self.app.timeframes
        }
        for tf, frame in frames.items():
            index = frame.candles.index
            results[tf].candles = frame.end - frame.start
            results[tf].start = index[frame.start].to_pydatetime()
            results[tf].end = index[frame.end - 1].to_pydatetime()

        pool, skipped = self._pool()
        signals = await self._replay_pool(pool, frames, progress)
        for tf, frame in frames.items():
            scores = []
            for strategy in pool:
                sig = signals.get((tf, strategy.name))
                if isinstance(sig, BaseException):
                    scores.append(self._failed(strategy, tf, sig))
                elif sig is not None:
                    scores.append(self._score(strategy, sig, frame, thresholds[tf], pip, tf))
            scores.sort(key=lambda s: (not s.ranked, -s.stats.score, -s.stats.signals))
            results[tf].strategies = scores
            results[tf].best = next((s.name for s in scores if s.ranked), None)
            results[tf].skipped = list(skipped)

        await progress("templates", 0, len(frames), "Searching rule templates")
        templates = template_candidates()
        for done, (tf, frame) in enumerate(frames.items(), start=1):
            results[tf].candidates = await asyncio.to_thread(
                self._search, templates, frame, thresholds[tf], pip
            )
            await progress("templates", done, len(frames), f"Searched templates on {tf}")

        spend: list[float] = []  # filled as LLM answers arrive, so a timeout keeps it
        if llm_rules and frames:
            await self._llm_stage(
                "LLM rule design",
                None,
                self._llm_rules(
                    resolved, frames, results, thresholds, pip, spread, progress, spend
                ),
                notes,
            )
        if llm_check and self.config.llm_top_n and frames:
            await self._llm_stage(
                "LLM check",
                "decision",
                self._llm_check(
                    resolved, frames, results, pool, signals, thresholds, progress, spend
                ),
                notes,
            )
        cost = sum(spend)

        for tf_result in results.values():
            tf_result.candidates.sort(key=_candidate_key)
            tf_result.best_candidate = next(
                (c.name for c in tf_result.candidates if self._robust(c)), None
            )
            if frames.get(tf_result.timeframe) and tf_result.best_candidate is None:
                tf_result.notes.append(
                    "No generated rule held up on the second half of the period."
                )

        result = BacktestResult(
            run_id=run_id,
            instrument=resolved,
            generated_at=datetime.now(UTC),
            duration_ms=int((time.perf_counter() - started) * 1000),
            days=self.config.days,
            window=self.window,
            pip_size=pip,
            spread_pips=round(spread / pip, 2) if spread else None,
            timeframes=results,
            llm_cost_usd=round(cost, 6) if cost else None,
            notes=notes,
        )
        await progress("done", 1, 1, "Backtest complete")
        await emit({"type": "backtest_complete", "result": result.model_dump(mode="json")})
        logger.info(
            "Backtest {} for {} finished in {} ms (best: {})",
            run_id,
            resolved,
            result.duration_ms,
            ", ".join(f"{tf}={r.best}" for tf, r in results.items()),
        )
        return result

    # ------------------------------------------------------------------ data

    async def _load(
        self, instrument: str, timeframes: Sequence[str] | None = None
    ) -> tuple[dict[str, _Frame], dict[str, str]]:
        timeframes = list(timeframes or self.app.timeframes)
        counts = [self.app.backtest_candles(tf) for tf in timeframes]
        fetched, spreads = await asyncio.gather(
            asyncio.gather(
                *(
                    self.source.get_candles(instrument, [tf], count=count)
                    for tf, count in zip(timeframes, counts, strict=True)
                ),
                return_exceptions=True,
            ),
            asyncio.gather(
                *(
                    self.source.get_spreads(instrument, tf, count=count)
                    for tf, count in zip(timeframes, counts, strict=True)
                ),
                return_exceptions=True,
            ),
        )
        frames: dict[str, _Frame] = {}
        errors: dict[str, str] = {}
        for tf, outcome, spread in zip(timeframes, fetched, spreads, strict=True):
            if isinstance(outcome, BaseException):
                errors[tf] = f"Could not load candles: {outcome}"
                continue
            candles = completed(outcome[tf])
            horizon = self.config.horizon_for(tf)
            end = len(candles) - horizon
            wanted = self.config.scored_bars(self.app.timeframes[tf].granularity_seconds)
            start = max(0, end - wanted)
            if end - start < 4 * horizon:
                errors[tf] = f"Not enough candles to backtest ({len(candles)} available)"
                continue
            moves = forward_moves(candles["close"].to_numpy(dtype=float), horizon)
            whole, train, validation = period_masks(len(candles), start, end)
            frames[tf] = _Frame(
                tf,
                candles,
                moves,
                whole,
                train,
                validation,
                start,
                end,
                horizon,
                _align_spreads(spread, candles.index),
            )
        return frames, errors

    async def _thresholds(
        self, instrument: str, frames: dict[str, _Frame]
    ) -> tuple[dict[str, Threshold], float | None, list[str]]:
        """What a hit must beat, per timeframe: the spread at each signal's own candle.
        Falls back to the live spread (or zero when the market is closed) only where
        OANDA's bid / ask candles were unavailable. Also returns the median spread."""
        thresholds: dict[str, Threshold] = {}
        notes: list[str] = []
        missing = [tf for tf, frame in frames.items() if frame.spreads is None]
        fallback: float | None = None
        if missing:
            fallback, note = await self._spread(instrument)
            if note:
                notes.append(note)
            elif len(missing) < len(frames):
                notes.append(f"Historical spreads unavailable for {', '.join(missing)}.")
        recorded: list[np.ndarray] = []
        for tf, frame in frames.items():
            if frame.spreads is None:
                thresholds[tf] = fallback or 0.0
            else:
                thresholds[tf] = frame.spreads
                recorded.append(frame.spreads[frame.start : frame.end])
        median = float(np.median(np.concatenate(recorded))) if recorded else fallback
        return thresholds, median, notes

    async def _pip_size(
        self, instrument: str, frames: dict[str, _Frame]
    ) -> tuple[float, str | None]:
        try:
            for info in await self.source.list_instruments():
                if info.name == instrument and info.pip_location is not None:
                    return 10.0**info.pip_location, None
        except Exception as exc:
            logger.debug("Instrument metadata unavailable: {}", exc)
        price = next((float(f.candles["close"].iloc[-1]) for f in frames.values()), 1.0)
        return (0.01 if price >= 20 else 0.0001), "Pip size inferred from the price level."

    async def _spread(self, instrument: str) -> tuple[float | None, str | None]:
        try:
            tick = await self.source.get_price(instrument)
        except Exception as exc:
            return None, (
                f"Live spread unavailable ({exc}): any move in the signalled direction "
                "counts as a hit."
            )
        if not tick.tradeable:
            return None, (
                "Market closed: the quoted spread is not representative, so any move in the "
                "signalled direction counts as a hit."
            )
        return tick.spread, None

    # ------------------------------------------------------------------ pool strategies

    def _pool(self) -> tuple[list[Strategy], list[SkippedStrategy]]:
        pool: list[Strategy] = []
        skipped: list[SkippedStrategy] = []
        for strategy in self.registry:
            if isinstance(strategy, IndicatorStrategy | RuleStrategy):
                pool.append(strategy)
            else:
                skipped.append(
                    SkippedStrategy(
                        name=strategy.name,
                        display_name=strategy.display_name,
                        reason="Only the LLM decides this strategy: there is no rule to replay.",
                    )
                )
        return pool, skipped

    async def _replay_pool(
        self,
        pool: Sequence[Strategy],
        frames: dict[str, _Frame],
        progress: Callable[[str, int, int, str], Awaitable[None]],
        *,
        through_last: bool = False,
        chunk_bars: int | None = None,
    ) -> dict[tuple[str, str], np.ndarray | BaseException]:
        """Signals per (timeframe, strategy) over the scored period, or up to the last
        candle with ``through_last`` (the final ``horizon`` candles cannot be judged but
        are worth drawing)."""
        signals: dict[tuple[str, str], np.ndarray | BaseException] = {}
        size = chunk_bars or self.config.chunk_bars
        jobs = [(tf, s) for tf in frames for s in pool]
        rules = [(tf, s) for tf, s in jobs if isinstance(s, RuleStrategy)]
        # Built-ins are split into chunks of candles so the load spreads evenly over the
        # worker processes (one 1m strategy alone is ~30 s of work).
        chunks: list[tuple[str, Strategy, int, int]] = []
        for tf, strategy in jobs:
            if isinstance(strategy, RuleStrategy):
                continue
            frame = frames[tf]
            last = len(frame.candles) if through_last else frame.end
            signals[(tf, strategy.name)] = np.full(len(frame.candles), np.nan)
            for first in range(frame.start, last, size):
                chunks.append((tf, strategy, first, min(first + size, last)))
        total = len(rules) + len(chunks)
        await progress("replay", 0, total, "Replaying strategies")

        done = 0
        for tf, strategy in rules:  # vectorised: milliseconds each
            params = self.registry.params_for(strategy.name, tf)
            try:
                signals[(tf, strategy.name)] = await asyncio.to_thread(
                    rule_signals, strategy, frames[tf].candles, params
                )
            except Exception as exc:
                signals[(tf, strategy.name)] = exc
            done += 1
        if not chunks:
            return signals

        loop = asyncio.get_running_loop()
        executor = self.executor_factory()

        async def replay(tf: str, strategy: Strategy, first: int, stop: int) -> tuple[Any, ...]:
            offset = max(0, first + 1 - self.window)  # the window of the chunk's first candle
            try:
                out = await loop.run_in_executor(
                    executor,
                    _replay_job,
                    strategy.name,
                    self.registry.params_for(strategy.name, tf),
                    frames[tf].candles.iloc[offset:stop],
                    self.window,
                    first - offset,
                    tf,
                )
                return tf, strategy.name, first, stop, out[first - offset :]
            except Exception as exc:
                return tf, strategy.name, first, stop, exc

        try:
            for next_done in asyncio.as_completed([replay(*chunk) for chunk in chunks]):
                tf, name, first, stop, outcome = await next_done
                current = signals[(tf, name)]
                if isinstance(outcome, BaseException):
                    signals[(tf, name)] = outcome
                elif isinstance(current, np.ndarray):
                    current[first:stop] = outcome
                done += 1
                await progress("replay", done, total, f"Replaying {name} on {tf}")
        finally:
            await asyncio.to_thread(executor.shutdown)
        return signals

    # ------------------------------------------------------------------ single replay

    async def replay_strategy(self, instrument: str, timeframe: str, name: str) -> StrategyTrace:
        """One strategy's last ``days`` on one timeframe, laid out for the chart."""
        started = time.perf_counter()
        if timeframe not in self.app.timeframes:
            configured = ", ".join(self.app.timeframes)
            raise ValueError(f"Unknown timeframe '{timeframe}'. Configured: {configured}")
        strategy = self.registry.get(name)
        if not isinstance(strategy, IndicatorStrategy | RuleStrategy):
            raise ValueError(
                f"'{strategy.display_name}' is decided by the LLM only: there is no rule to replay."
            )
        resolved = await self.source.resolve_instrument(instrument)
        frames, errors = await self._load(resolved, [timeframe])
        frame = frames.get(timeframe)
        if frame is None:
            raise ValueError(errors.get(timeframe) or f"No {timeframe} candles for {resolved}")
        pip, pip_note = await self._pip_size(resolved, frames)
        thresholds, spread, spread_notes = await self._thresholds(resolved, frames)

        # one strategy: split its candles so every worker gets a share
        chunk = max(100, math.ceil((len(frame.candles) - frame.start) / self.workers))
        signals = await self._replay_pool(
            [strategy], frames, _quiet, through_last=True, chunk_bars=chunk
        )
        sig = signals[(timeframe, name)]
        if isinstance(sig, BaseException):
            raise RuntimeError(f"Replay of {name} on {timeframe} failed: {sig}") from sig
        data = build_trace(
            candles=frame.candles,
            signals=sig,
            moves=frame.moves,
            start=frame.start,
            end=frame.end,
            horizon=frame.horizon,
            threshold=thresholds[timeframe],
            pip=pip,
        )
        trace = StrategyTrace(
            **dict(data),
            instrument=resolved,
            timeframe=timeframe,
            strategy=name,
            display_name=strategy.display_name,
            origin="rule" if isinstance(strategy, RuleStrategy) else "builtin",
            category=strategy.category,
            description=strategy.description,
            params=self.registry.params_for(name, timeframe),
            generated_at=datetime.now(UTC),
            duration_ms=int((time.perf_counter() - started) * 1000),
            horizon_bars=frame.horizon,
            pip_size=pip,
            spread_pips=round(spread / pip, 2) if spread else None,
            notes=[n for n in (pip_note, *spread_notes) if n],
        )
        logger.info(
            "Replayed {} on {} {} in {} ms ({} entries)",
            name,
            resolved,
            timeframe,
            trace.duration_ms,
            len(trace.entries),
        )
        return trace

    def _score(
        self,
        strategy: Strategy,
        sig: np.ndarray,
        frame: _Frame,
        threshold: Threshold,
        pip: float,
        timeframe: str,
        origin: StrategyOrigin | None = None,
    ) -> StrategyScore:
        def score(mask: np.ndarray) -> HitStats:
            return score_signals(sig, frame.moves, threshold=threshold, pip=pip, select=mask)

        stats = score(frame.whole)
        return StrategyScore(
            name=strategy.name,
            display_name=strategy.display_name,
            origin=origin or ("rule" if isinstance(strategy, RuleStrategy) else "builtin"),
            category=strategy.category,
            description=strategy.description,
            params=(
                self.registry.params_for(strategy.name, timeframe)
                if strategy.name in self.registry
                else {}
            ),
            stats=stats,
            train=score(frame.train),
            validation=score(frame.validation),
            ranked=stats.signals >= self.config.min_signals,
        )

    def _failed(self, strategy: Strategy, timeframe: str, exc: BaseException) -> StrategyScore:
        logger.warning("Backtest replay of {} on {} failed: {}", strategy.name, timeframe, exc)
        return StrategyScore(
            name=strategy.name,
            display_name=strategy.display_name,
            origin="rule" if isinstance(strategy, RuleStrategy) else "builtin",
            category=strategy.category,
            description=strategy.description,
            error=f"{type(exc).__name__}: {exc}",
        )

    # ------------------------------------------------------------------ new strategies

    @property
    def _half_min(self) -> int:
        return max(3, math.ceil(self.config.min_signals / 2))

    def _robust(self, candidate: StrategyScore) -> bool:
        """Held up out of sample: ``min_signals`` on the second half and better than a
        coin flip on both halves."""
        train, validation = candidate.train, candidate.validation
        return bool(
            train
            and validation
            and validation.signals >= self.config.min_signals
            and (train.hit_rate or 0) > 0.5
            and (validation.hit_rate or 0) > 0.5
        )

    def _candidate(
        self,
        rule: RuleStrategy,
        frame: _Frame,
        threshold: Threshold,
        pip: float,
        origin: StrategyOrigin,
    ) -> StrategyScore:
        sig = rule_signals(rule, frame.candles, {})
        scored = self._score(rule, sig, frame, threshold, pip, frame.timeframe, origin)
        scored.ranked = bool(scored.validation and scored.validation.signals >= self._half_min)
        scored.rule = rule.spec.model_dump(mode="json", exclude_none=True)
        return scored

    def _search(
        self, templates: Sequence[RuleStrategy], frame: _Frame, threshold: Threshold, pip: float
    ) -> list[StrategyScore]:
        """Pick finalists on the first half, judge them on the second."""
        trained: list[tuple[float, RuleStrategy]] = []
        for rule in templates:
            sig = rule_signals(rule, frame.candles, {})
            train = score_signals(
                sig, frame.moves, threshold=threshold, pip=pip, select=frame.train
            )
            if train.signals >= self._half_min:
                trained.append((train.score, rule))
        trained.sort(key=lambda item: -item[0])
        finalists = [
            self._candidate(rule, frame, threshold, pip, "template")
            for _, rule in trained[:TEMPLATE_FINALISTS]
        ]
        finalists.sort(key=_candidate_key)
        return finalists[:TEMPLATES_SHOWN]

    async def _llm_stage(
        self,
        label: str,
        task: Literal["decision"] | None,
        work: Coroutine[Any, Any, None],
        notes: list[str],
    ) -> None:
        """Run an optional LLM stage with cloud models only, within the time budget."""
        router = self.router()
        probe = LLMRequest(messages=[Message.user("probe")], task=task, remote_only=True)
        usable = [
            p for p in router.chain("text", probe) if router.state[p.name].healthy is not False
        ]
        if not usable:
            work.close()
            notes.append(
                f"{label} skipped: no cloud model in the text role is available (local models "
                "are too slow for a backtest). Fix an API key or pick a cloud model."
            )
            return
        budget = self.config.llm_timeout_seconds
        try:
            async with asyncio.timeout(budget):
                await work
        except TimeoutError:
            notes.append(f"{label} stopped after {budget:.0f} s; partial results are shown.")

    async def _llm_rules(
        self,
        instrument: str,
        frames: dict[str, _Frame],
        results: dict[str, TimeframeBacktest],
        thresholds: dict[str, Threshold],
        pip: float,
        spread: float | None,
        progress: Callable[[str, int, int, str], Awaitable[None]],
        spend: list[float],
    ) -> None:
        total, done = len(frames), 0
        await progress("llm_rules", 0, total, "Asking the LLM for new rules")

        async def design(tf: str, frame: _Frame) -> None:
            nonlocal done
            result = results[tf]
            tested = [_tested_line(s) for s in (result.strategies + result.candidates)[:12]]
            try:
                rules, provider, spent = await design_rules(
                    self.router(),
                    instrument=instrument,
                    timeframe=tf,
                    market=_market_lines(frame, pip, spread),
                    tested=tested,
                    count=self.config.llm_rule_candidates,
                )
            except (LLMError, ValueError) as exc:
                result.notes.append(f"LLM rule design failed: {str(exc)[:300]}")
            else:
                spend.append(spent or 0.0)
                for rule in rules:
                    result.candidates.append(
                        await asyncio.to_thread(
                            self._candidate, rule, frame, thresholds[tf], pip, "llm"
                        )
                    )
                result.notes.append(f"{len(rules)} rule(s) proposed by {provider}.")
            done += 1
            await progress("llm_rules", done, total, f"LLM rules for {tf}")

        await asyncio.gather(*(design(tf, frame) for tf, frame in frames.items()))

    # ------------------------------------------------------------------ LLM check

    async def _llm_check(
        self,
        instrument: str,
        frames: dict[str, _Frame],
        results: dict[str, TimeframeBacktest],
        pool: Sequence[Strategy],
        signals: dict[tuple[str, str], np.ndarray | BaseException],
        thresholds: dict[str, Threshold],
        progress: Callable[[str, int, int, str], Awaitable[None]],
        spend: list[float],
    ) -> None:
        by_name = {s.name: s for s in pool}
        plans: list[tuple[str, StrategyScore, Strategy, np.ndarray, list[int]]] = []
        for tf, frame in frames.items():
            top = [s for s in results[tf].strategies if s.ranked][: self.config.llm_top_n]
            for scored in top:
                sig = signals.get((tf, scored.name))
                if not isinstance(sig, np.ndarray):
                    continue
                candidates = np.flatnonzero(
                    frame.validation & ~np.isnan(frame.moves) & (np.nan_to_num(sig) != 0)
                )
                if not len(candidates):
                    continue
                count = min(self.config.llm_samples, len(candidates))
                picks = np.linspace(0, len(candidates) - 1, count).round().astype(int)
                rows = [int(candidates[i]) for i in dict.fromkeys(picks.tolist())]
                plans.append((tf, scored, by_name[scored.name], sig, rows))

        total = sum(len(rows) for *_, rows in plans)
        if not total:
            return
        done = 0
        await progress("llm_check", 0, total, "Checking top strategies with the LLM")
        context = StrategyContext(
            instrument=instrument,
            history_length=self.app.analysis.history_length,
            prompt_candles=self.app.analysis.prompt_candles,
            document_prompt_candles=self.app.analysis.document_prompt_candles,
            chart_candles=self.app.analysis.chart_candles,
        )

        async def sample(tf: str, strategy: Strategy, row: int) -> Any:
            nonlocal done
            frame = frames[tf]
            view = frame.candles.iloc[max(0, row + 1 - self.window) : row + 1]
            try:
                params = self.registry.params_for(strategy.name, tf)
                features = await strategy.compute(view, params, tf, context=context)
                decision = (
                    await self.decisions.decide(features, instrument, view, remote_only=True)
                    if features.status == "ok"
                    else None
                )
            except Exception as exc:
                logger.debug("LLM check sample failed: {}", exc)
                decision = None
            done += 1
            await progress("llm_check", done, total, f"LLM check {strategy.name} on {tf}")
            return decision

        for tf, scored, strategy, sig, rows in plans:
            decisions = await asyncio.gather(*(sample(tf, strategy, row) for row in rows))
            check = LLMCheck(samples=len(rows))
            agree = rule_hits = 0
            spent: list[float] = []
            for row, decision in zip(rows, decisions, strict=True):
                move, rule_dir = frames[tf].moves[row], float(np.sign(sig[row]))
                threshold = threshold_at(thresholds[tf], row)
                rule_hits += int(rule_dir * move > threshold)
                if decision is None or decision.decided_by == "none":
                    check.errors += 1
                    continue
                if decision.cost_usd is not None:
                    spent.append(decision.cost_usd)
                if decision.decided_by not in check.providers:
                    check.providers.append(decision.decided_by)
                llm_dir = SIGNAL_CODES[decision.signal]
                agree += int(llm_dir == rule_dir)
                if llm_dir == 0:
                    check.abstained += 1
                    continue
                check.llm_signals += 1
                check.llm_hits += int(llm_dir * move > threshold)
            answered = check.samples - check.errors
            check.llm_hit_rate = _ratio(check.llm_hits, check.llm_signals)
            check.rule_hit_rate = _ratio(rule_hits, check.samples)
            check.agreement = _ratio(agree, answered)
            check.cost_usd = round(sum(spent), 6) if spent else None
            spend.append(sum(spent))
            scored.llm_check = check


# --------------------------------------------------------------------------- helpers


def _replay_job(
    name: str, params: dict[str, Any], candles: pd.DataFrame, window: int, start: int, tf: str
) -> np.ndarray:
    """Process-pool entry point (positional arguments pickle cleanly)."""
    return replay_builtin(name, params, candles, window=window, start=start, timeframe=tf)


def _align_spreads(spread: pd.Series | BaseException, index: pd.Index) -> np.ndarray | None:
    """Spreads on the candles' times (gaps filled from neighbours); None if unusable."""
    if isinstance(spread, BaseException) or spread.empty:
        if isinstance(spread, BaseException):
            logger.debug("Historical spreads unavailable: {}", spread)
        return None
    aligned = spread.reindex(index).ffill().bfill().to_numpy(dtype=float)
    return None if np.isnan(aligned).any() else np.maximum(aligned, 0.0)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _candidate_key(candidate: StrategyScore) -> tuple[bool, float, int]:
    validation = candidate.validation or HitStats()
    return (not candidate.ranked, -validation.score, -validation.signals)


def _tested_line(scored: StrategyScore) -> str:
    stats = scored.validation if scored.origin in ("template", "llm") else scored.stats
    if stats is None or not stats.signals:
        return f"{scored.display_name} [{scored.name}]: no BUY/SELL signals"
    return (
        f"{scored.display_name} [{scored.name}] ({scored.category}): "
        f"{(stats.hit_rate or 0):.0%} over {stats.signals} signals"
    )


def _market_lines(frame: _Frame, pip: float, spread: float | None) -> list[str]:
    candles = frame.candles
    period = candles.iloc[frame.start : frame.end + frame.horizon]
    close = period["close"]
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    trend = ema(candles["close"], 50).iloc[frame.start : frame.end + frame.horizon]
    above = float((close > trend).mean())
    true_range = float(atr(candles, 14).iloc[frame.start : frame.end].mean())
    lines = [
        f"Backtest period: {period.index[0]:%Y-%m-%d %H:%M} to {period.index[-1]:%Y-%m-%d %H:%M}"
        f" UTC ({len(period)} candles). Signals are judged {frame.horizon} candles later.",
        f"Price moved from {first:.6g} to {last:.6g} ({(last / first - 1):+.2%}); range "
        f"{float(period['low'].min()):.6g} - {float(period['high'].max()):.6g} "
        f"({(float(period['high'].max()) - float(period['low'].min())) / pip:.0f} pips).",
        f"Average true range: {true_range / pip:.1f} pips per candle; close above its 50-EMA "
        f"{above:.0%} of the time.",
    ]
    if spread:
        lines.append(f"Spread: {spread / pip:.1f} pips (a hit must beat it).")
    return lines


def rule_file(result: BacktestResult, timeframe: str, name: str) -> tuple[str, str]:
    """``(strategy name, YAML text)`` for saving a generated candidate as a rule strategy."""
    tf_result = result.timeframes.get(timeframe)
    candidate = next(
        (c for c in (tf_result.candidates if tf_result else []) if c.name == name), None
    )
    if candidate is None or candidate.rule is None:
        raise BacktestNotFoundError(f"No candidate '{name}' for {timeframe} in this backtest")
    instrument = result.instrument.lower()
    strategy_name = f"bt_{instrument}_{timeframe}_{name}"[:40].rstrip("_")
    validation = candidate.validation or HitStats()
    start = tf_result.start if tf_result else None
    end = tf_result.end if tf_result else None
    period = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}" if start and end else "the backtest period"
    spec = dict(candidate.rule)
    spec.update(
        name=strategy_name,
        display_name=f"{candidate.display_name} ({result.instrument} {timeframe})",
        description=(
            f"{candidate.description} Backtested on {result.instrument} {timeframe}: "
            f"{(validation.hit_rate or 0):.0%} hit rate over {validation.signals} signals "
            f"in the second half of {period}."
        ),
    )
    load_rule(spec, strategy_name)  # validate before anything is written
    header = (
        f"# Generated by the TradeSense backtest ({candidate.origin}) on "
        f"{result.generated_at:%Y-%m-%d %H:%M} UTC for {result.instrument} {timeframe}.\n"
        "# Past hit rates over a few days are a small sample, not a promise.\n"
    )
    return strategy_name, header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
