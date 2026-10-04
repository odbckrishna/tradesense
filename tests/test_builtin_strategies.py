from __future__ import annotations

import math
import warnings

import pytest

from tests.helpers import REPO_ROOT, make_candles, trending_candles
from tradesense.strategies.base import StrategyContext
from tradesense.strategies.registry import StrategyRegistry, discover_builtins

BUILTINS = sorted(discover_builtins())
TIMEFRAMES = ["1m", "5m", "15m"]


@pytest.fixture(scope="module")
def registry() -> StrategyRegistry:
    reg = StrategyRegistry(
        rules_dir=REPO_ROOT / "config" / "rules",
        strategies_config_path=REPO_ROOT / "config" / "strategies.yml",
    )
    reg.reload()
    return reg


def test_twelve_builtins_registered() -> None:
    assert BUILTINS == [
        "adx_trend",
        "atr_breakout",
        "bollinger_squeeze",
        "ema_crossover",
        "ichimoku",
        "macd_momentum",
        "pivot_points",
        "rsi_divergence",
        "rsi_extremes",
        "stochastic_crossover",
        "supertrend",
        "vwap_bounce",
    ]


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
@pytest.mark.parametrize("name", BUILTINS)
async def test_compute_on_every_timeframe(
    registry: StrategyRegistry, name: str, timeframe: str
) -> None:
    df = make_candles(299, timeframe, seed=11)
    strategy = registry.get(name)
    params = registry.params_for(name, timeframe)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no pandas/numpy warnings allowed
        features = await strategy.compute(
            df, params, timeframe, context=StrategyContext(history_length=10)
        )

    assert features.status == "ok", features.message
    assert features.strategy == name
    assert features.timeframe == timeframe
    assert features.source == "builtin"
    assert features.pre_signal in {"BUY", "SELL", "SIDEWAYS"}
    assert features.pre_signal_reason
    assert features.events
    assert features.latest
    assert features.candles_available == 299
    assert features.last_candle_time == df.index[-1].to_pydatetime()
    for key, history in features.history.items():
        assert len(history) == 10, key
    numeric = [v for v in features.latest.values() if isinstance(v, float)]
    assert numeric
    assert all(math.isfinite(v) for v in numeric)


@pytest.mark.parametrize("name", BUILTINS)
async def test_insufficient_data_is_reported_not_raised(
    registry: StrategyRegistry, name: str
) -> None:
    strategy = registry.get(name)
    df = make_candles(12, "5m")
    features = await strategy.compute(df, registry.params_for(name, "5m"), "5m")
    assert features.status == "insufficient_data"
    assert features.candles_available == 12
    assert features.candles_required > 12
    assert "Needs at least" in (features.message or "") or "period" in (features.message or "")
    assert features.pre_signal is None


async def test_params_drive_required_candles(registry: StrategyRegistry) -> None:
    ema = registry.get("ema_crossover")
    df = make_candles(150, "15m")
    # 15m override is fast=50, slow=200 -> needs 220 candles
    features = await ema.compute(df, registry.params_for("ema_crossover", "15m"), "15m")
    assert features.status == "insufficient_data"
    assert features.candles_required == 220
    ok = await ema.compute(df, registry.params_for("ema_crossover", "5m"), "5m")
    assert ok.status == "ok"
    assert ok.params["fast"] == 9


@pytest.mark.parametrize("name", ["ema_crossover", "supertrend", "adx_trend", "ichimoku"])
@pytest.mark.parametrize(("direction", "expected"), [("up", "BUY"), ("down", "SELL")])
async def test_trend_strategies_follow_clear_trends(
    registry: StrategyRegistry, name: str, direction: str, expected: str
) -> None:
    df = trending_candles(direction)
    features = await registry.get(name).compute(df, registry.params_for(name, "5m"), "5m")
    assert features.pre_signal == expected, features.pre_signal_reason


async def test_invalid_parameters_raise_strategy_error(registry: StrategyRegistry) -> None:
    from tradesense.strategies.base import StrategyError

    with pytest.raises(StrategyError, match="fast"):
        await registry.get("ema_crossover").compute(
            make_candles(200), {"fast": 30, "slow": 10}, "5m"
        )


async def test_events_mention_candle_offsets(registry: StrategyRegistry) -> None:
    features = await registry.get("macd_momentum").compute(make_candles(299, seed=5), {}, "1m")
    assert any("candle" in event for event in features.events)
