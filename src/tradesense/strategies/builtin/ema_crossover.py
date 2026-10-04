"""EMA Crossover (trend)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradesense.indicators.technical import (
    ago,
    atr,
    bars_since,
    crossed_above,
    crossed_below,
    ema,
    last_valid,
)
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, StrategyError, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class EmaCrossover(IndicatorStrategy):
    name = "ema_crossover"
    display_name = "EMA Crossover"
    description = (
        "Trend following: a fast EMA crossing above (below) a slow EMA marks a bullish "
        "(bearish) trend change; price position relative to both EMAs confirms it."
    )
    category = "trend"
    min_candles = 30
    default_params = {"fast": 9, "slow": 21, "lookback": 20}  # noqa: RUF012

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["slow"]) + int(params["lookback"]))

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        fast_n, slow_n = int(params["fast"]), int(params["slow"])
        lookback = int(params["lookback"])
        if fast_n >= slow_n:
            raise StrategyError("parameter 'fast' must be smaller than 'slow'")
        close = df["close"]
        fast, slow = ema(close, fast_n), ema(close, slow_n)
        f, s, a = require(
            ema_fast=last_valid(fast), ema_slow=last_valid(slow), atr=last_valid(atr(df, 14))
        )
        c = float(close.iloc[-1])

        fb.series("ema_fast", fast)
        fb.series("ema_slow", slow)
        fb.series("close", close)
        spread = fast - slow
        fb.series("spread", spread)
        spread_atr = (f - s) / a
        fb.value("spread_atr", spread_atr)
        fb.value("atr", a)

        up = bars_since(crossed_above(fast, slow), lookback)
        down = bars_since(crossed_below(fast, slow), lookback)
        if up is not None:
            fb.event(f"Bullish crossover: EMA{fast_n} crossed above EMA{slow_n} {ago(up)}")
        if down is not None:
            fb.event(f"Bearish crossover: EMA{fast_n} crossed below EMA{slow_n} {ago(down)}")
        if up is None and down is None:
            fb.event(f"No crossover in the last {lookback} candles")
        if c > max(f, s):
            fb.event("Price is above both EMAs")
        elif c < min(f, s):
            fb.event("Price is below both EMAs")
        else:
            fb.event("Price is between the EMAs")
        earlier = last_valid(spread.iloc[:-3].abs())
        if earlier is not None:
            fb.event(
                "EMAs are diverging (trend strengthening)"
                if abs(f - s) > earlier
                else "EMAs are converging (trend weakening)"
            )

        if abs(spread_atr) < 0.1:
            fb.signal("SIDEWAYS", f"EMAs are intertwined (gap {spread_atr:+.2f} ATR)")
        elif f > s:
            if c >= s:
                fb.signal(
                    "BUY", f"EMA{fast_n} above EMA{slow_n} and price holds above the slow EMA"
                )
            else:
                fb.signal("SIDEWAYS", "Bullish EMA order but price fell below the slow EMA")
        elif c <= s:
            fb.signal("SELL", f"EMA{fast_n} below EMA{slow_n} and price holds below the slow EMA")
        else:
            fb.signal("SIDEWAYS", "Bearish EMA order but price rose above the slow EMA")
