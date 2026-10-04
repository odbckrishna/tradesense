"""Supertrend (trend)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradesense.indicators.technical import ago, atr, bars_since, last_valid, supertrend
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class Supertrend(IndicatorStrategy):
    name = "supertrend"
    display_name = "Supertrend"
    description = (
        "ATR-based trailing band: price above the Supertrend line is an up-trend (BUY), below "
        "is a down-trend (SELL). Fresh flips mark trend changes."
    )
    category = "trend"
    min_candles = 30
    default_params = {"period": 10, "multiplier": 3.0}  # noqa: RUF012

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["period"]) + 20)

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        period = int(params["period"])
        st = supertrend(df, period, float(params["multiplier"]))
        line, direction, a = require(
            supertrend=last_valid(st["value"]),
            direction=last_valid(st["direction"]),
            atr=last_valid(atr(df, period)),
        )
        c = float(df["close"].iloc[-1])
        distance_atr = (c - line) / a
        fb.series("close", df["close"])
        fb.series("supertrend", st["value"])
        fb.value("direction", "up" if direction > 0 else "down")
        fb.value("distance_atr", distance_atr)

        dirs = st["direction"]
        flips = (dirs != dirs.shift(1)) & dirs.shift(1).notna()
        flip = bars_since(flips, 100)
        trend = "UP" if direction > 0 else "DOWN"
        if flip is not None:
            fb.event(f"Supertrend flipped to {trend} {ago(flip)}")
        else:
            fb.event(f"Supertrend has been {trend} for over 100 candles")
        fb.event(
            f"Price is {abs(distance_atr):.2f} ATR {'above' if c > line else 'below'} the line"
        )

        fresh = " (fresh flip)" if flip is not None and flip <= 2 else ""
        if direction > 0:
            fb.signal("BUY", f"Supertrend is up: price above the trailing band{fresh}")
        else:
            fb.signal("SELL", f"Supertrend is down: price below the trailing band{fresh}")
