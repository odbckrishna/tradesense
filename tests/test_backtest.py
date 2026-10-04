"""Backtest: scoring, live/backtest signal consistency, templates, LLM rules, engine."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.apptools import build_services
from tests.fakes import VALID_JSON, ScriptedProvider, make_router
from tests.helpers import REPO_ROOT, make_candles
from tradesense.backtest.designer import parse_rule_designs
from tradesense.backtest.engine import BacktestBusyError, Backtester, _replay_job, rule_file
from tradesense.backtest.scoring import (
    forward_moves,
    period_masks,
    score_signals,
    wilson_lower,
)
from tradesense.backtest.signals import SIGNAL_CODES, replay_builtin, rule_signals
from tradesense.backtest.templates import template_candidates
from tradesense.backtest.trace import build_trace
from tradesense.config import BacktestConfig, Settings
from tradesense.data.models import InstrumentInfo, PriceTick
from tradesense.llm.decision import DecisionEngine
from tradesense.llm.messages import LLMRequest, Message
from tradesense.llm.schemas import DecisionParseError
from tradesense.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyFeatures,
    UnknownStrategyError,
)
from tradesense.strategies.builtin.ema_crossover import EmaCrossover
from tradesense.strategies.builtin.rsi_extremes import RsiExtremes
from tradesense.strategies.rule_engine import load_rule, load_rule_file

PIP = 0.0001
NAN = np.nan


# --------------------------------------------------------------------------- scoring


def test_forward_moves_and_masks() -> None:
    close = np.array([1.0, 1.1, 1.3, 1.2, 1.5])
    np.testing.assert_allclose(forward_moves(close, 2)[:3], [0.3, 0.1, 0.2])
    assert np.isnan(forward_moves(close, 2)[3:]).all()  # the future is unknown: never scored
    whole, train, validation = period_masks(10, 2, 8)
    assert whole.tolist() == [False, False, *[True] * 6, False, False]
    assert train.nonzero()[0].tolist() == [2, 3, 4]
    assert validation.nonzero()[0].tolist() == [5, 6, 7]


def test_score_signals_counts_hits_against_the_spread() -> None:
    signals = np.array([1, 1, -1, 0, NAN, 1, 1])
    moves = np.array([0.002, -0.001, -0.003, 0.0005, 0.001, 0.0001, NAN])
    stats = score_signals(signals, moves, threshold=0.0002, pip=PIP)
    assert (stats.signals, stats.buys, stats.sells, stats.sideways) == (4, 3, 1, 1)
    # +0.0001 is a favourable move but smaller than the spread: a miss
    assert (stats.hits, stats.buy_hits, stats.sell_hits) == (2, 1, 1)
    assert stats.hit_rate == 0.5
    assert stats.buy_hit_rate == pytest.approx(1 / 3, abs=1e-4)
    assert stats.sell_hit_rate == 1.0
    assert stats.episodes == 3  # BUY run, SELL, BUY again after SIDEWAYS
    assert stats.avg_move_pips == pytest.approx((20 - 10 + 30 + 1) / 4, abs=0.01)
    selected = score_signals(signals, moves, threshold=0.0002, pip=PIP, select=np.arange(7) < 2)
    assert (selected.signals, selected.hits) == (2, 1)


def test_wilson_bound_prefers_large_samples() -> None:
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(3, 3) < wilson_lower(70, 100)
    assert 0.6 < wilson_lower(70, 100) < 0.7


# --------------------------------------------------------------------------- consistency


async def test_rule_signals_match_the_live_rule() -> None:
    rule = load_rule_file(REPO_ROOT / "config" / "rules" / "rsi_ema_combo.yml")
    df = make_candles(260, "5m", seed=3)
    signals = rule_signals(rule, df, {})
    assert np.isnan(signals[: rule.required_candles({}) - 1]).all()
    for i in range(rule.required_candles({}) - 1, len(df), 17):
        features = await rule.compute(df.iloc[: i + 1], {}, "5m")
        assert SIGNAL_CODES[features.pre_signal] == signals[i], i  # type: ignore[index]


async def test_builtin_replay_matches_live_compute() -> None:
    df = make_candles(220, "1m", seed=5)
    strategy = EmaCrossover()
    params = dict(strategy.default_params)
    signals = replay_builtin("ema_crossover", params, df, window=120, start=100, timeframe="1m")
    assert np.isnan(signals[:100]).all()
    for i in range(100, len(df), 13):
        live = await strategy.compute(df.iloc[max(0, i + 1 - 120) : i + 1], params, "1m")
        assert SIGNAL_CODES[live.pre_signal] == signals[i], i  # type: ignore[index]


def test_replay_runs_in_a_process_pool() -> None:
    df = make_candles(80, "1m", seed=1)
    with ProcessPoolExecutor(max_workers=1) as pool:
        out = pool.submit(_replay_job, "rsi_extremes", {}, df, 60, 60, "1m").result()
    assert out.shape == (80,)
    assert not np.isnan(out[60:]).all()


# --------------------------------------------------------------------------- new strategies


def test_templates_are_valid_unique_and_short() -> None:
    candidates = template_candidates()
    names = [c.name for c in candidates]
    assert len(names) == len(set(names)) > 80
    assert max(map(len, names)) <= 16  # leaves room for "bt_<instrument>_<tf>_"


def test_parse_rule_designs() -> None:
    text = json.dumps(
        {
            "rules": [
                {
                    "name": "Fast EMA momentum!",
                    "description": "EMA slope with RSI",
                    "category": "momentum",
                    "indicators": [
                        {"id": "ema", "type": "ema", "period": 8},
                        {"id": "rsi", "type": "rsi", "period": 9},
                    ],
                    "buy": "rising(ema) and rsi > 55",
                    "sell": "falling(ema) and rsi < 45",
                },
                {
                    "name": "bad",
                    "category": "trend",
                    "indicators": [{"id": "x", "type": "ichimoku_cloud"}],
                    "buy": "x > 1",
                    "sell": "x < 1",
                },
            ]
        }
    )
    rules = parse_rule_designs(text, limit=3)
    assert [r.name for r in rules] == ["llm_fast_ema_mom"]  # 16 chars max
    assert rules[0].display_name == "Fast EMA momentum!"
    with pytest.raises(DecisionParseError, match="no valid rule"):
        parse_rule_designs(json.dumps({"rules": [json.loads(text)["rules"][1]]}), 3)
    with pytest.raises(DecisionParseError):
        parse_rule_designs("no json here", 3)
    with pytest.raises(DecisionParseError, match="non-empty list"):
        parse_rule_designs('{"rules": []}', 3)


# --------------------------------------------------------------------------- engine


class FakeSource:
    def __init__(self) -> None:
        self.counts: dict[str, int | None] = {}

    async def resolve_instrument(self, raw: str) -> str:
        return raw.upper()

    async def get_candles(
        self, instrument: str, timeframes: list[str], count: int | None = None
    ) -> dict[str, pd.DataFrame]:
        tf = timeframes[0]
        self.counts[tf] = count
        return {tf: make_candles(count or 300, tf, seed=11, last_incomplete=True)}

    async def get_price(self, instrument: str) -> PriceTick:
        # a wide live spread (e.g. the Sunday open) that must NOT be used for history
        return PriceTick(
            instrument=instrument, time=datetime.now(UTC), bid=1.1, ask=1.1005, tradeable=True
        )

    async def get_spreads(
        self, instrument: str, timeframe: str, count: int | None = None
    ) -> pd.Series:
        candles = make_candles(count or 300, timeframe, seed=11, last_incomplete=True)
        return pd.Series(0.00008, index=candles.index)

    async def list_instruments(self) -> list[InstrumentInfo]:
        return [InstrumentInfo(name="EUR_USD", pip_location=-4)]


class DocumentStub(Strategy):
    name = "doc_strategy"
    display_name = "Uploaded doc"
    description = "LLM only"
    source = "document"
    category = "pattern"

    async def analyze(
        self, df: pd.DataFrame, params: dict[str, Any], timeframe: str, context: StrategyContext
    ) -> StrategyFeatures:
        raise AssertionError("never replayed")


class PoolStub:
    def __init__(self, *strategies: Strategy) -> None:
        self._strategies = {s.name: s for s in strategies}

    def __iter__(self) -> Any:
        return iter(self._strategies.values())

    def __contains__(self, name: object) -> bool:
        return name in self._strategies

    def params_for(self, name: str, timeframe: str) -> dict[str, Any]:
        return dict(self._strategies[name].default_params)

    def get(self, name: str) -> Strategy:
        try:
            return self._strategies[name]
        except KeyError:
            raise UnknownStrategyError(name) from None


LLM_RULES = json.dumps(
    {
        "rules": [
            {
                "name": "macd zero",
                "description": "MACD line above zero",
                "category": "momentum",
                "indicators": [{"id": "m", "type": "macd"}],
                "buy": "m > 0",
                "sell": "m < 0",
            }
        ]
    }
)


def make_backtester(settings: Settings, source: FakeSource) -> tuple[Backtester, ScriptedProvider]:
    app = settings.app.model_copy(
        update={
            "timeframes": {tf: settings.app.timeframes[tf] for tf in ("5m", "15m")},
            "backtest": BacktestConfig(days=1, min_signals=6, llm_top_n=2, llm_samples=3),
            "oanda": settings.app.oanda.model_copy(update={"candle_count": 120}),
        }
    )
    jev = ScriptedProvider("jev", [VALID_JSON], capabilities=["decision"])
    writer = ScriptedProvider("writer", [LLM_RULES])
    router = make_router([jev, writer], roles={"text": ["jev", "writer"]})
    rule = load_rule_file(REPO_ROOT / "config" / "rules" / "rsi_ema_combo.yml")
    backtester = Backtester(
        source=source,
        registry=PoolStub(EmaCrossover(), RsiExtremes(), rule, DocumentStub()),  # type: ignore[arg-type]
        decisions=DecisionEngine(router),
        router=lambda: router,
        app=app,
        executor_factory=lambda: ThreadPoolExecutor(max_workers=2),
    )
    return backtester, jev


async def test_backtester_end_to_end(settings: Settings) -> None:
    source = FakeSource()
    backtester, jev = make_backtester(settings, source)
    events: list[dict[str, Any]] = []

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    result = await backtester.run("eur_usd", on_event=collect)

    assert result.instrument == "EUR_USD"
    assert result.pip_size == pytest.approx(0.0001)
    assert result.spread_pips == pytest.approx(0.8)
    assert source.counts == {"5m": 288 + 120 + 6, "15m": 96 + 120 + 4}
    tf = result.timeframes["5m"]
    assert tf.candles == 288
    assert tf.horizon_bars == 6
    assert {s.name for s in tf.strategies} == {"ema_crossover", "rsi_extremes", "rsi_ema_combo"}
    assert [s.name for s in tf.skipped] == ["doc_strategy"]
    ranked = [s for s in tf.strategies if s.ranked]
    assert tf.best == ranked[0].name
    assert [s.stats.score for s in ranked] == sorted((s.stats.score for s in ranked), reverse=True)
    assert all(s.train and s.validation for s in tf.strategies)

    origins = {c.origin for c in tf.candidates}
    assert origins == {"template", "llm"}
    llm_rule = next(c for c in tf.candidates if c.origin == "llm")
    assert llm_rule.name == "llm_macd_zero"
    assert llm_rule.rule
    assert any("proposed by writer" in note for note in tf.notes)

    checked = [s for s in ranked[:2] if s.llm_check]
    assert checked
    assert all(1 <= s.llm_check.samples <= 3 for s in checked)
    check = checked[0].llm_check
    assert check is not None
    assert check.providers == ["jev"]
    assert check.errors == 0
    assert check.llm_signals == check.samples  # VALID_JSON is a BUY
    assert all(r.task == "decision" for r in jev.calls)

    kinds = [e["type"] for e in events]
    assert kinds[0] == "backtest_started"
    assert kinds[-1] == "backtest_complete"
    stages = {e["stage"] for e in events if e["type"] == "backtest_progress"}
    assert stages == {"candles", "replay", "templates", "llm_rules", "llm_check", "done"}

    name, text = rule_file(result, "5m", tf.candidates[0].name)
    assert name.startswith("bt_eur_usd_5m_")
    assert len(name) <= 40
    assert text.startswith("# Generated by the TradeSense backtest")
    assert load_rule(yaml.safe_load(text)).name == name


async def test_backtester_without_llm_and_short_history(settings: Settings) -> None:
    class ShortSource(FakeSource):
        async def get_candles(
            self, instrument: str, timeframes: list[str], count: int | None = None
        ) -> dict[str, pd.DataFrame]:
            tf = timeframes[0]
            return {tf: make_candles(10 if tf == "15m" else (count or 300), tf, seed=2)}

        async def get_price(self, instrument: str) -> PriceTick:
            raise RuntimeError("no account")

        async def get_spreads(
            self, instrument: str, timeframe: str, count: int | None = None
        ) -> pd.Series:
            raise RuntimeError("no bid/ask candles")

    backtester, jev = make_backtester(settings, ShortSource())
    result = await backtester.run("EUR_USD", llm_check=False, llm_rules=False)
    assert result.spread_pips is None
    assert any("spread" in n for n in result.notes)
    assert result.timeframes["15m"].error is not None
    assert not result.timeframes["15m"].strategies
    assert result.timeframes["5m"].strategies
    assert jev.calls == []  # no LLM spend when both LLM options are off


async def test_one_backtest_at_a_time(tmp_path: Path) -> None:
    services, _ = build_services(tmp_path)
    await services.backtest_lock.acquire()
    try:
        with pytest.raises(BacktestBusyError):
            await services.run_backtest("EUR_USD")
    finally:
        services.backtest_lock.release()


class LocalScripted(ScriptedProvider):
    @property
    def is_local(self) -> bool:
        return True


def test_remote_only_requests_skip_local_providers() -> None:
    local = LocalScripted("local", [VALID_JSON])
    cloud = ScriptedProvider("cloud", [VALID_JSON])
    router = make_router([local, cloud], roles={"text": ["local", "cloud"]})
    request = LLMRequest(messages=[Message.user("x")], remote_only=True)
    assert [p.name for p in router.chain("text", request)] == ["cloud"]
    assert [
        p.name for p in router.chain("text", request.model_copy(update={"remote_only": False}))
    ] == [
        "local",
        "cloud",
    ]


async def test_llm_stages_skip_local_models_and_respect_the_budget(settings: Settings) -> None:
    backtester, _ = make_backtester(settings, FakeSource())
    local = LocalScripted("local", [LLM_RULES])
    backtester.router = lambda: make_router([local], roles={"text": ["local"]})
    result = await backtester.run("EUR_USD")
    assert local.calls == []  # never sent to a slow local model
    skipped = [n for n in result.notes if "skipped" in n]
    assert len(skipped) == 2  # rule design and LLM check

    slow = ScriptedProvider("slow", [LLM_RULES], delay=5)
    backtester.router = lambda: make_router([slow], roles={"text": ["slow"]})
    backtester.config = backtester.config.model_copy(
        update={"llm_timeout_seconds": 0.2, "llm_top_n": 0}
    )
    result = await backtester.run("EUR_USD")
    assert any("stopped after 0 s" in n for n in result.notes)
    assert all(c.origin == "template" for c in result.timeframes["5m"].candidates)


# --------------------------------------------------------------------------- replay


def test_build_trace_marks_entries_and_outcomes() -> None:
    idx = pd.date_range("2026-10-01", periods=10, freq="5min", tz="UTC")
    close = np.array([1.0, 1.0, 1.001, 1.003, 1.0012, 1.0, 0.998, 0.999, 1.0, 1.001])
    candles = pd.DataFrame(
        {"open": close, "high": close + 0.0005, "low": close - 0.0005, "close": close},
        index=idx,
    ).assign(volume=10.0)
    signals = np.array([NAN, NAN, 1, 1, 0, -1, -1, 1, -1, -1])
    moves = forward_moves(close, 2)
    trace = build_trace(
        candles=candles,
        signals=signals,
        moves=moves,
        start=1,
        end=8,
        horizon=2,
        threshold=0.0005,
        pip=PIP,
    )
    assert len(trace.candles) == len(trace.signals) == 9  # from the first scored candle
    assert trace.candles[0].time == int(idx[1].timestamp())
    assert trace.signals[:3] == [None, 1, 1]
    entries = [(e.signal, e.candles, e.hit) for e in trace.entries]
    # BUY moved +2 pips (< the 5 pip spread: a miss), SELL and BUY moved 10 / 20 pips the
    # right way, the last SELL is too recent to judge
    assert entries == [("BUY", 2, False), ("SELL", 2, True), ("BUY", 1, True), ("SELL", 2, None)]
    first = trace.entries[0]
    assert first.judged_time == int(idx[4].timestamp())
    assert first.judged_price == pytest.approx(1.0012)
    assert first.move_pips == pytest.approx(2.0)
    assert trace.entries[3].judged_time is None
    stats = trace.entry_stats
    assert (stats.entries, stats.hits, stats.pending) == (3, 2, 1)
    assert (stats.buy_entries, stats.buy_hits, stats.sell_entries, stats.sell_hits) == (2, 1, 1, 1)
    assert stats.hit_rate == pytest.approx(2 / 3, abs=1e-4)
    scored = np.zeros(10, dtype=bool)
    scored[1:8] = True
    assert trace.stats == score_signals(signals, moves, threshold=0.0005, pip=PIP, select=scored)


async def test_replay_strategy_matches_the_backtest(settings: Settings) -> None:
    source = FakeSource()
    backtester, jev = make_backtester(settings, source)
    trace = await backtester.replay_strategy("eur_usd", "5m", "ema_crossover")
    assert trace.instrument == "EUR_USD"
    assert trace.origin == "builtin"
    assert trace.horizon_bars == 6
    # the scored day plus the last 6 candles, which are drawn but cannot be judged yet
    assert len(trace.candles) == len(trace.signals) == 288 + 6
    times = [c.time for c in trace.candles]
    for entry in trace.entries:
        i = times.index(entry.time)
        assert trace.signals[i] == (1 if entry.signal == "BUY" else -1)
        assert i == 0 or trace.signals[i - 1] != trace.signals[i]
        assert (entry.hit is None) == (i >= 288)
    result = await backtester.run("eur_usd", llm_check=False, llm_rules=False)
    scored = next(s for s in result.timeframes["5m"].strategies if s.name == "ema_crossover")
    assert trace.stats == scored.stats  # same candles, same rules, same numbers
    assert jev.calls == []

    rule = await backtester.replay_strategy("EUR_USD", "15m", "rsi_ema_combo")
    assert rule.origin == "rule"
    assert len(rule.candles) == 96 + 4


async def test_replay_strategy_errors(settings: Settings) -> None:
    backtester, _ = make_backtester(settings, FakeSource())
    with pytest.raises(ValueError, match="LLM only"):
        await backtester.replay_strategy("EUR_USD", "5m", "doc_strategy")
    with pytest.raises(ValueError, match="Unknown timeframe"):
        await backtester.replay_strategy("EUR_USD", "4h", "ema_crossover")
    with pytest.raises(UnknownStrategyError):
        await backtester.replay_strategy("EUR_USD", "5m", "nope")


def test_score_signals_with_a_spread_per_candle() -> None:
    signals = np.array([1.0, 1.0, -1.0])
    moves = np.array([0.0003, 0.0003, -0.0003])
    # the same 3-pip move beats a 1-pip spread but not a 5-pip one
    spreads = np.array([0.0001, 0.0005, 0.0001])
    stats = score_signals(signals, moves, threshold=spreads, pip=PIP)
    assert (stats.hits, stats.buy_hits, stats.sell_hits) == (2, 1, 1)
    assert score_signals(signals, moves, threshold=0.0005, pip=PIP).hits == 0


async def test_hits_are_judged_against_historical_spreads(settings: Settings) -> None:
    """The live quote (5 pips here) must not judge two days of history."""
    backtester, _ = make_backtester(settings, FakeSource())
    result = await backtester.run("EUR_USD", llm_check=False, llm_rules=False)
    assert result.spread_pips == pytest.approx(0.8)  # median of the bid/ask candles
    assert not any("spread" in note.lower() for note in result.notes)
    trace = await backtester.replay_strategy("EUR_USD", "5m", "ema_crossover")
    assert trace.spread_pips == pytest.approx(0.8)
    judged = [e for e in trace.entries if e.hit is not None]
    assert judged
    for entry in judged:
        assert entry.move_pips is not None
        if abs(entry.move_pips - 0.8) > 0.01:  # move_pips is rounded to 0.01
            assert entry.hit == (entry.move_pips > 0.8)
