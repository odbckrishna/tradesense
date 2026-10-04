"""Per-candle signals of pool strategies, exactly as the live app would have produced them.

Built-in strategies are replayed candle by candle on the same ``window`` of completed
candles the live app analyses (their pre-signal is computed from the latest candle only).
That costs ~5-10 ms per candle, so :func:`replay_builtin` is a top-level function a
process pool can run. Rule strategies are vectorised and cost milliseconds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy
from tradesense.strategies.registry import discover_builtins
from tradesense.strategies.rule_engine import RuleStrategy

SIGNAL_CODES = {"BUY": 1.0, "SELL": -1.0, "SIDEWAYS": 0.0}


def replay_builtin(
    name: str,
    params: dict[str, Any],
    candles: pd.DataFrame,
    *,
    window: int,
    start: int,
    timeframe: str,
) -> np.ndarray:
    """Pre-signal of built-in ``name`` at every candle from ``start`` on, each computed from
    the ``window`` candles ending at that candle. NaN where the strategy cannot run."""
    strategy = discover_builtins()[name]()
    if not isinstance(strategy, IndicatorStrategy):
        raise TypeError(f"'{name}' is not an indicator strategy")
    merged = strategy.merge_params(params)
    needed = strategy.required_candles(merged)
    out = np.full(len(candles), np.nan)
    for i in range(max(start, needed - 1), len(candles)):
        view = candles.iloc[max(0, i + 1 - window) : i + 1]
        if len(view) < needed:
            continue
        builder = FeatureBuilder(strategy, view, merged, timeframe, history_length=1)
        try:
            strategy.evaluate(view, merged, builder)
        except Exception:  # warming up or degenerate data: no signal on this candle
            continue
        if builder.pre_signal is not None:
            out[i] = SIGNAL_CODES[builder.pre_signal]
    return out


def rule_signals(
    strategy: RuleStrategy, candles: pd.DataFrame, params: dict[str, Any]
) -> np.ndarray:
    """Vectorised signal of a YAML rule strategy at every candle."""
    return strategy.signal_series(candles, params).to_numpy(dtype=float)
