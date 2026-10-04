"""Ichimoku Cloud (trend)."""

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
    ichimoku,
    last_valid,
)
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class Ichimoku(IndicatorStrategy):
    name = "ichimoku"
    display_name = "Ichimoku Cloud"
    description = (
        "Price above the cloud with Tenkan above Kijun is bullish; below the cloud with Tenkan "
        "below Kijun is bearish; inside the cloud is indecision."
    )
    category = "trend"
    min_candles = 80
    default_params = {"conversion": 9, "base": 26, "span_b": 52, "displacement": 26}  # noqa: RUF012

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["span_b"]) + int(params["displacement"]) + 2)

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        lines = ichimoku(
            df,
            int(params["conversion"]),
            int(params["base"]),
            int(params["span_b"]),
            int(params["displacement"]),
        )
        tenkan, kijun, span_a, span_b, future_a, future_b, a = require(
            tenkan=last_valid(lines["tenkan"]),
            kijun=last_valid(lines["kijun"]),
            senkou_a=last_valid(lines["senkou_a"]),
            senkou_b=last_valid(lines["senkou_b"]),
            future_a=last_valid(lines["future_a"]),
            future_b=last_valid(lines["future_b"]),
            atr=last_valid(atr(df, 14)),
        )
        c = float(df["close"].iloc[-1])
        top, bottom = max(span_a, span_b), min(span_a, span_b)
        fb.series("close", df["close"])
        fb.series("tenkan", lines["tenkan"])
        fb.series("kijun", lines["kijun"])
        fb.series("senkou_a", lines["senkou_a"], history=False)
        fb.series("senkou_b", lines["senkou_b"], history=False)
        fb.value("cloud_top", top)
        fb.value("cloud_bottom", bottom)
        fb.value("cloud_thickness_atr", (top - bottom) / a)
        fb.series("chikou_delta", lines["chikou_delta"], history=False)

        position = "above" if c > top else "below" if c < bottom else "inside"
        fb.event(f"Price is {position} the cloud")
        up = bars_since(crossed_above(lines["tenkan"], lines["kijun"]), 10)
        down = bars_since(crossed_below(lines["tenkan"], lines["kijun"]), 10)
        if up is not None:
            fb.event(f"Bullish TK cross (Tenkan above Kijun) {ago(up)}")
        if down is not None:
            fb.event(f"Bearish TK cross (Tenkan below Kijun) {ago(down)}")
        fb.event(f"Future cloud is {'bullish' if future_a > future_b else 'bearish'}")
        chikou = last_valid(lines["chikou_delta"])
        if chikou is not None:
            fb.event(
                f"Chikou span is {'above' if chikou > 0 else 'below'} price from "
                f"{int(params['displacement'])} candles ago"
            )

        if position == "above" and tenkan > kijun:
            fb.signal("BUY", "Price above the cloud with Tenkan above Kijun")
        elif position == "below" and tenkan < kijun:
            fb.signal("SELL", "Price below the cloud with Tenkan below Kijun")
        elif position == "inside":
            fb.signal("SIDEWAYS", "Price inside the cloud: indecision")
        else:
            fb.signal("SIDEWAYS", "Cloud and Tenkan/Kijun disagree")
