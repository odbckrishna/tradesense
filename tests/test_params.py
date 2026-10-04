from __future__ import annotations

from tradesense.config import Settings, StrategyParamsConfig
from tradesense.strategies.base import resolve_params
from tradesense.strategies.builtin.ema_crossover import EmaCrossover
from tradesense.strategies.builtin.pivot_points import PivotPoints


def test_resolution_order_override_then_params_then_defaults() -> None:
    cfg = StrategyParamsConfig(
        params={"fast": 9, "slow": 21},
        overrides={"15m": {"fast": 50, "slow": 200}, "1m": {"slow": 30}},
    )
    strategy = EmaCrossover()
    assert resolve_params(strategy, cfg, "15m") == {"fast": 50, "slow": 200, "lookback": 20}
    assert resolve_params(strategy, cfg, "1m") == {"fast": 9, "slow": 30, "lookback": 20}
    assert resolve_params(strategy, cfg, "5m") == {"fast": 9, "slow": 21, "lookback": 20}
    assert resolve_params(strategy, None, "5m") == EmaCrossover.default_params


def test_unknown_and_invalid_params_are_ignored() -> None:
    cfg = StrategyParamsConfig(params={"fast": "nine", "speed": 3, "slow": 25.0})
    resolved = resolve_params(EmaCrossover(), cfg, "5m")
    assert resolved == {"fast": 9, "slow": 25, "lookback": 20}
    assert isinstance(resolved["slow"], int)


def test_string_params_are_coerced() -> None:
    cfg = StrategyParamsConfig(params={"period": "4h"}, overrides={"15m": {"period": "1D"}})
    assert resolve_params(PivotPoints(), cfg, "15m")["period"] == "1D"
    assert resolve_params(PivotPoints(), cfg, "1m")["period"] == "4h"


def test_repo_strategies_yml(settings: Settings) -> None:
    cfg = settings.strategies.for_strategy("pivot_points")
    assert resolve_params(PivotPoints(), cfg, "1m")["period"] == "1h"
    assert resolve_params(PivotPoints(), cfg, "15m")["period"] == "1D"
    assert resolve_params(PivotPoints(), cfg, "5m")["period"] == "4h"
