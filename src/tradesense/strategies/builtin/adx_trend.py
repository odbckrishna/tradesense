"""ADX Trend Strength (+DI / -DI)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradesense.indicators.technical import (
    adx,
    ago,
    bars_since,
    crossed_above,
    crossed_below,
    last_valid,
)
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class AdxTrend(IndicatorStrategy):
    name = "adx_trend"
    display_name = "ADX Trend Strength"
    description = (
        "ADX above the threshold indicates a trending market; +DI above -DI says the trend is "
        "up (BUY), -DI above +DI says down (SELL). Weak ADX means range (SIDEWAYS)."
    )
    category = "trend"
    min_candles = 40
    default_params = {"period": 14, "threshold": 25.0, "lookback": 10}  # noqa: RUF012

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["period"]) * 3)

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        threshold, lookback = float(params["threshold"]), int(params["lookback"])
        lines = adx(df, int(params["period"]))
        value, plus, minus, earlier = require(
            adx=last_valid(lines["adx"]),
            plus_di=last_valid(lines["plus_di"]),
            minus_di=last_valid(lines["minus_di"]),
            adx_3_ago=last_valid(lines["adx"].iloc[:-3]),
        )
        fb.series("adx", lines["adx"])
        fb.series("plus_di", lines["plus_di"])
        fb.series("minus_di", lines["minus_di"])
        fb.series("close", df["close"], history=False)
        slope = value - earlier
        fb.value("adx_change_3", slope)

        strength = "strong" if value >= threshold else "weak"
        fb.event(f"ADX {value:.1f}: {strength} trend (threshold {threshold:g})")
        fb.event(f"ADX is {'rising' if slope > 0 else 'falling'} ({slope:+.1f} over 3 candles)")
        up = bars_since(crossed_above(lines["plus_di"], lines["minus_di"]), lookback)
        down = bars_since(crossed_below(lines["plus_di"], lines["minus_di"]), lookback)
        if up is not None:
            fb.event(f"+DI crossed above -DI {ago(up)}")
        if down is not None:
            fb.event(f"-DI crossed above +DI {ago(down)}")
        fb.event(f"+DI {plus:.1f} vs -DI {minus:.1f}")

        if value >= threshold and plus > minus:
            fb.signal("BUY", "Strong trend with +DI leading")
        elif value >= threshold and minus > plus:
            fb.signal("SELL", "Strong trend with -DI leading")
        else:
            fb.signal("SIDEWAYS", "ADX below threshold: no strong trend")
