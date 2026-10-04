"""Stochastic Crossover (momentum)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradesense.indicators.technical import (
    ago,
    bars_since,
    crossed_above,
    crossed_below,
    last_valid,
    stochastic,
)
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, StrategyError, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class StochasticCrossover(IndicatorStrategy):
    name = "stochastic_crossover"
    display_name = "Stochastic Crossover"
    description = (
        "%K crossing above %D inside the oversold zone is bullish; crossing below %D inside the "
        "overbought zone is bearish."
    )
    category = "momentum"
    min_candles = 30
    default_params = {  # noqa: RUF012
        "k": 14,
        "d": 3,
        "smooth": 3,
        "overbought": 80.0,
        "oversold": 20.0,
        "lookback": 3,
    }

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(
            self.min_candles,
            int(params["k"]) + int(params["d"]) + int(params["smooth"]) + 10,
        )

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        overbought, oversold = float(params["overbought"]), float(params["oversold"])
        lookback = int(params["lookback"])
        if not 0 < oversold < overbought < 100:
            raise StrategyError("require 0 < oversold < overbought < 100")
        lines = stochastic(df, int(params["k"]), int(params["d"]), int(params["smooth"]))
        k_line, d_line = lines["k"], lines["d"]
        k, d = require(k=last_valid(k_line), d=last_valid(d_line))
        fb.series("stoch_k", k_line)
        fb.series("stoch_d", d_line)
        fb.series("close", df["close"], history=False)

        prev_k = k_line.shift(1)
        in_oversold = (prev_k <= oversold) | (d_line <= oversold)
        in_overbought = (prev_k >= overbought) | (d_line >= overbought)
        bull = bars_since(crossed_above(k_line, d_line) & in_oversold, lookback)
        bear = bars_since(crossed_below(k_line, d_line) & in_overbought, lookback)
        any_up = bars_since(crossed_above(k_line, d_line), 10)
        any_down = bars_since(crossed_below(k_line, d_line), 10)

        zone = "oversold" if k <= oversold else "overbought" if k >= overbought else "neutral"
        fb.event(f"%K at {k:.1f} ({zone} zone), %D at {d:.1f}")
        if bull is not None:
            fb.event(f"%K crossed above %D in the oversold zone {ago(bull)}")
        elif any_up is not None:
            fb.event(f"%K crossed above %D outside the oversold zone {ago(any_up)}")
        if bear is not None:
            fb.event(f"%K crossed below %D in the overbought zone {ago(bear)}")
        elif any_down is not None:
            fb.event(f"%K crossed below %D outside the overbought zone {ago(any_down)}")

        if bull is not None and (bear is None or bull < bear):
            fb.signal("BUY", "Bullish %K/%D crossover from oversold")
        elif bear is not None:
            fb.signal("SELL", "Bearish %K/%D crossover from overbought")
        else:
            fb.signal("SIDEWAYS", "No stochastic crossover in an extreme zone")
