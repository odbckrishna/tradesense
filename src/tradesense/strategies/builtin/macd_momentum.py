"""MACD Momentum."""

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
    macd,
)
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, StrategyError, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class MacdMomentum(IndicatorStrategy):
    name = "macd_momentum"
    display_name = "MACD Momentum"
    description = (
        "Momentum: MACD above its signal line with a rising histogram is bullish; below with a "
        "falling histogram is bearish. Zero-line crosses confirm the trend."
    )
    category = "momentum"
    min_candles = 45
    default_params = {"fast": 12, "slow": 26, "signal": 9, "lookback": 20}  # noqa: RUF012

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["slow"]) + int(params["signal"]) + 10)

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        fast, slow = int(params["fast"]), int(params["slow"])
        lookback = int(params["lookback"])
        if fast >= slow:
            raise StrategyError("parameter 'fast' must be smaller than 'slow'")
        lines = macd(df["close"], fast, slow, int(params["signal"]))
        m, s, h, h_prev = require(
            macd=last_valid(lines["macd"]),
            signal=last_valid(lines["signal"]),
            hist=last_valid(lines["hist"]),
            hist_previous=last_valid(lines["hist"].iloc[:-1]),
        )
        fb.series("macd", lines["macd"])
        fb.series("signal", lines["signal"])
        fb.series("hist", lines["hist"])
        fb.series("close", df["close"], history=False)

        up = bars_since(crossed_above(lines["macd"], lines["signal"]), lookback)
        down = bars_since(crossed_below(lines["macd"], lines["signal"]), lookback)
        zero_up = bars_since(crossed_above(lines["macd"], 0.0), lookback)
        zero_down = bars_since(crossed_below(lines["macd"], 0.0), lookback)
        if up is not None:
            fb.event(f"MACD crossed above its signal line {ago(up)}")
        if down is not None:
            fb.event(f"MACD crossed below its signal line {ago(down)}")
        if zero_up is not None:
            fb.event(f"MACD crossed above zero {ago(zero_up)}")
        if zero_down is not None:
            fb.event(f"MACD crossed below zero {ago(zero_down)}")
        rising = h > h_prev
        hist = lines["hist"].dropna()
        streak = 1
        for i in range(len(hist) - 1, 1, -1):
            if (hist.iloc[i] > hist.iloc[i - 1]) != rising:
                break
            streak += 1
        fb.event(f"Histogram {'rising' if rising else 'falling'} for {streak} candles")
        fb.event(f"MACD is {'above' if m > 0 else 'below'} the zero line")

        if m > s and rising:
            fb.signal("BUY", "MACD above signal with an expanding histogram")
        elif m < s and not rising:
            fb.signal("SELL", "MACD below signal with an expanding negative histogram")
        elif m > s:
            fb.signal("SIDEWAYS", "Bullish MACD but momentum is fading")
        else:
            fb.signal("SIDEWAYS", "Bearish MACD but momentum is fading")
