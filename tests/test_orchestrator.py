from __future__ import annotations

import time
from typing import Any

import pandas as pd
import pytest

from tests.fakes import VALID_JSON, ScriptedProvider, make_router
from tests.helpers import REPO_ROOT, make_candles
from tradesense.config import LLMSettings, TimeframeConfig
from tradesense.data.oanda_client import InstrumentNotFoundError, OandaServerError
from tradesense.engine import auto_refresh
from tradesense.engine import orchestrator as orchestrator_module
from tradesense.engine.auto_refresh import AutoRefresher, next_run_delay
from tradesense.engine.orchestrator import Orchestrator
from tradesense.llm.decision import DecisionEngine
from tradesense.strategies.base import (
    FeatureBuilder,
    ImageAttachment,
    Strategy,
    StrategyContext,
    StrategyFeatures,
)
from tradesense.strategies.registry import StrategyRegistry

TIMEFRAMES = {
    "1m": TimeframeConfig(granularity="M1", refresh_seconds=60),
    "5m": TimeframeConfig(granularity="M5", refresh_seconds=300),
    "15m": TimeframeConfig(granularity="M15", refresh_seconds=900),
}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class FakeSource:
    def __init__(self, frames: dict[str, pd.DataFrame], fail: set[str] | None = None) -> None:
        self.frames = frames
        self.fail = fail or set()
        self.requests: list[list[str]] = []

    async def resolve_instrument(self, raw: str) -> str:
        if raw.upper().replace("/", "_") != "EUR_USD":
            raise InstrumentNotFoundError(raw, ["EUR_USD"])
        return "EUR_USD"

    async def get_candles(
        self, instrument: str, timeframes: list[str], count: int | None = None
    ) -> dict[str, pd.DataFrame]:
        self.requests.append(timeframes)
        for tf in timeframes:
            if tf in self.fail:
                raise OandaServerError(f"OANDA server error (503) for {tf}", 503)
        return {tf: self.frames[tf] for tf in timeframes}


class BrokenStrategy(Strategy):
    name = "broken_strategy"
    display_name = "Broken"
    description = "always fails"
    source = "rule"
    category = "pattern"
    min_candles = 5

    async def analyze(
        self, df: pd.DataFrame, params: dict[str, Any], timeframe: str, context: StrategyContext
    ) -> StrategyFeatures:
        raise RuntimeError("kaboom")


class PackStrategy(Strategy):
    """Uses the shared feature pack and attaches a fake live chart."""

    description = "uses feature pack"
    source = "image"
    category = "pattern"
    min_candles = 5

    def __init__(self, name: str) -> None:
        self.name = name
        self.display_name = name

    async def analyze(
        self, df: pd.DataFrame, params: dict[str, Any], timeframe: str, context: StrategyContext
    ) -> StrategyFeatures:
        assert context.feature_pack is not None
        pack = context.feature_pack()
        fb = FeatureBuilder(self, df, params, timeframe)
        fb.latest.update(pack["latest"])
        fb.images = [
            ImageAttachment.from_bytes("reference", PNG, "image/png"),
            ImageAttachment.from_bytes("live_chart", PNG, "image/png"),
        ]
        return fb.build()


def frames(n: int = 299) -> dict[str, pd.DataFrame]:
    return {tf: make_candles(n, tf, last_incomplete=True) for tf in TIMEFRAMES}


def build(
    source: FakeSource,
    *,
    delay: float = 0.0,
    extra: list[Strategy] | None = None,
    script: list[str | Exception] | None = None,
) -> tuple[Orchestrator, ScriptedProvider]:
    settings = LLMSettings(max_concurrency=32, retry_backoff_seconds=0)
    text = ScriptedProvider(
        "llm",
        script or [VALID_JSON],
        delay=delay,
        settings=settings,
        capabilities=["text", "vision"],
    )
    router = make_router([text], roles={"text": ["llm"], "vision": ["llm"]}, settings=settings)
    registry = StrategyRegistry(
        rules_dir=REPO_ROOT / "config" / "rules",
        strategies_config_path=REPO_ROOT / "config" / "strategies.yml",
        reference_loader=lambda: (list(extra or []), []),
    )
    registry.reload()
    orchestrator = Orchestrator(
        source=source,
        registry=registry,
        decisions=DecisionEngine(router),
        timeframes=TIMEFRAMES,
    )
    return orchestrator, text


SELECTIONS = {
    "1m": ["ema_crossover", "rsi_extremes"],
    "5m": ["ema_crossover", "macd_momentum", "rsi_ema_combo"],
    "15m": ["ichimoku", "rsi_ema_combo"],
}


async def test_all_pairs_run_in_parallel_and_stream() -> None:
    orchestrator, provider = build(FakeSource(frames()), delay=0.3)
    events: list[dict[str, Any]] = []

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    started = time.perf_counter()
    result = await orchestrator.run_analysis("eur/usd", SELECTIONS, on_event=collect)
    elapsed = time.perf_counter() - started

    assert result.instrument == "EUR_USD"
    assert len(provider.calls) == 7
    assert elapsed < 0.3 * 7 / 2, f"pairs ran sequentially ({elapsed:.2f}s)"
    assert events[0]["type"] == "analysis_started"
    assert events[-1]["type"] == "analysis_complete"
    results = [e for e in events if e["type"] == "strategy_result"]
    assert len(results) == 7
    consensus = [e for e in events if e["type"] == "consensus"]
    assert {e["timeframe"] for e in consensus} == {"1m", "5m", "15m"}
    for event in consensus:  # each consensus follows all results of its timeframe
        position = events.index(event)
        tf_results = [
            i
            for i, e in enumerate(events)
            if e["type"] == "strategy_result" and e["timeframe"] == event["timeframe"]
        ]
        assert max(tf_results) < position

    tf5 = result.timeframes["5m"]
    assert [s.strategy for s in tf5.strategies] == [
        "ema_crossover",
        "macd_momentum",
        "rsi_ema_combo",
    ]
    assert all(s.status == "ok" for s in tf5.strategies)
    assert tf5.consensus is not None
    assert tf5.consensus.signal == "BUY"
    assert tf5.candles == 298  # in-progress candle excluded from analysis
    assert result.usage.decisions == 7
    assert result.usage.total_tokens == 7 * 150
    assert result.usage.total_cost_usd == pytest.approx(7 * 200 / 1e6)
    assert tf5.strategies[0].params == {"fast": 9, "slow": 21, "lookback": 20}
    assert result.timeframes["15m"].strategies[0].decision is not None


async def test_errors_are_isolated_per_slot() -> None:
    orchestrator, _ = build(FakeSource(frames()), extra=[BrokenStrategy()])
    result = await orchestrator.run_analysis(
        "EUR_USD", {"5m": ["ema_crossover", "broken_strategy", "does_not_exist", "supertrend"]}
    )
    by_name = {s.strategy: s for s in result.timeframes["5m"].strategies}
    assert by_name["ema_crossover"].status == "ok"
    assert by_name["supertrend"].status == "ok"
    assert by_name["broken_strategy"].status == "error"
    assert "kaboom" in (by_name["broken_strategy"].message or "")
    assert by_name["does_not_exist"].status == "error"
    assert "Unknown strategy" in (by_name["does_not_exist"].message or "")
    assert result.timeframes["5m"].consensus is not None


async def test_failed_timeframe_only_affects_its_column() -> None:
    orchestrator, _ = build(FakeSource(frames(), fail={"1m"}))
    result = await orchestrator.run_analysis("EUR_USD", SELECTIONS)
    one = result.timeframes["1m"]
    assert one.error is not None
    assert "503" in one.error
    assert all(s.status == "error" for s in one.strategies)
    assert all(s.status == "ok" for s in result.timeframes["5m"].strategies)


async def test_unknown_timeframe_is_an_error_entry() -> None:
    orchestrator, _ = build(FakeSource(frames()))
    result = await orchestrator.run_analysis("EUR_USD", {"2h": ["ema_crossover"], "5m": []})
    assert "Unknown timeframe" in (result.timeframes["2h"].error or "")
    assert result.timeframes["5m"].strategies == []


async def test_insufficient_data_skips_the_llm() -> None:
    short = {tf: make_candles(25, tf) for tf in TIMEFRAMES}
    orchestrator, provider = build(FakeSource(short))
    result = await orchestrator.run_analysis("EUR_USD", {"1m": ["ichimoku", "ema_crossover"]})
    statuses = {s.strategy: s for s in result.timeframes["1m"].strategies}
    assert statuses["ichimoku"].status == "insufficient_data"
    assert statuses["ichimoku"].candles_required == 80
    assert statuses["ichimoku"].candles_available == 25
    assert "Needs at least 80" in (statuses["ichimoku"].message or "")
    assert not provider.calls
    assert result.timeframes["1m"].consensus is None


async def test_unknown_instrument_fails_the_run() -> None:
    orchestrator, _ = build(FakeSource(frames()))
    with pytest.raises(InstrumentNotFoundError):
        await orchestrator.run_analysis("FOO_BAR", {"5m": ["ema_crossover"]})


async def test_feature_pack_once_per_timeframe_and_live_chart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = orchestrator_module.compute_feature_pack

    def counting(df: pd.DataFrame, history_length: int = 10) -> dict[str, Any]:
        calls.append(len(df))
        return original(df, history_length)

    monkeypatch.setattr(orchestrator_module, "compute_feature_pack", counting)
    orchestrator, provider = build(
        FakeSource(frames()), extra=[PackStrategy("pack_a"), PackStrategy("pack_b")]
    )
    result = await orchestrator.run_analysis(
        "EUR_USD", {"5m": ["pack_a", "pack_b"], "15m": ["pack_a"]}
    )
    assert len(calls) == 2  # once per timeframe, shared by both strategies
    item = result.timeframes["5m"].strategies[0]
    assert item.reference_image_url == "/api/library/pack_a/file"
    assert item.live_chart_url is not None
    chart_id = item.live_chart_url.rsplit("/", 1)[-1].removesuffix(".png")
    assert orchestrator.charts.get(chart_id) == PNG
    assert all(call.has_images for call in provider.calls)


async def test_broken_event_consumer_does_not_break_analysis() -> None:
    orchestrator, _ = build(FakeSource(frames()))

    async def broken(event: dict[str, Any]) -> None:
        raise ConnectionError("websocket closed")

    result = await orchestrator.run_analysis("EUR_USD", {"5m": ["ema_crossover"]}, on_event=broken)
    assert result.timeframes["5m"].strategies[0].status == "ok"


async def test_llm_failure_degrades_to_sideways_not_error() -> None:
    orchestrator, _ = build(FakeSource(frames()), script=["garbage"])
    result = await orchestrator.run_analysis("EUR_USD", {"5m": ["ema_crossover"]})
    item = result.timeframes["5m"].strategies[0]
    assert item.status == "ok"
    assert item.decision is not None
    assert item.decision.signal == "SIDEWAYS"
    assert item.decision.confidence == 0
    assert result.timeframes["5m"].consensus is None  # fallback decisions are excluded


# --------------------------------------------------------------------------- auto refresh


def test_next_run_delay_aligns_to_candle_close() -> None:
    assert next_run_delay(60, 3, now=1000.5) == pytest.approx(22.5)
    assert next_run_delay(300, 0, now=1200.0) == pytest.approx(300.0)
    assert next_run_delay(900, 2, now=899.0) == pytest.approx(3.0)


async def test_auto_refresh_reruns_each_timeframe(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    monkeypatch.setattr(auto_refresh, "next_run_delay", lambda *a, **k: 0.01)
    orchestrator, _ = build(FakeSource(frames()))
    events: list[dict[str, Any]] = []

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    refresher = AutoRefresher(orchestrator, TIMEFRAMES, collect)
    refresher.start("EUR_USD", {"1m": ["ema_crossover"], "5m": ["supertrend"], "15m": []})
    for _ in range(200):
        await asyncio.sleep(0.02)
        completed = [e for e in events if e["type"] == "analysis_complete"]
        if len(completed) >= 4:
            break
    assert refresher.active
    status = refresher.status()
    assert set(status["next_runs"]) == {"1m", "5m"}
    refresher.stop()
    await asyncio.sleep(0.05)
    assert not refresher.active
    timeframes = {
        tf for e in events if e["type"] == "analysis_complete" for tf in e["result"]["timeframes"]
    }
    assert timeframes == {"1m", "5m"}
