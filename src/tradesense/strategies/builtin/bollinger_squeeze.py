"""Bollinger Band Squeeze Breakout (volatility)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tradesense.indicators.technical import ago, bars_since, bbands, last_valid
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class BollingerSqueeze(IndicatorStrategy):
    name = "bollinger_squeeze"
    display_name = "Bollinger Squeeze Breakout"
    description = (
        "Volatility compression (Bollinger bandwidth in its lowest percentile) followed by a "
        "close outside the bands signals a breakout in that direction."
    )
    category = "volatility"
    min_candles = 60
    default_params = {  # noqa: RUF012
        "period": 20,
        "std": 2.0,
        "squeeze_percentile": 20.0,
        "lookback": 120,
        "breakout_window": 5,
    }

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["period"]) + 40)

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        period, lookback = int(params["period"]), int(params["lookback"])
        window = int(params["breakout_window"])
        pct = float(params["squeeze_percentile"])
        bands = bbands(df["close"], period, float(params["std"]))
        close = df["close"]
        width = bands["width"]
        middle, w = require(middle=last_valid(bands["middle"]), width=last_valid(width))
        recent_width = width.dropna().iloc[-lookback:]
        threshold = float(np.percentile(recent_width, pct))
        width_rank = float((recent_width <= w).mean() * 100)

        fb.series("close", close)
        fb.series("bb_upper", bands["upper"])
        fb.series("bb_middle", bands["middle"])
        fb.series("bb_lower", bands["lower"])
        fb.series("bandwidth_pct", width)
        fb.series("percent_b", bands["percent_b"], history=False)
        fb.value("bandwidth_percentile", width_rank)
        fb.value("squeeze_threshold", threshold)

        squeeze_mask = width <= threshold
        squeeze_now = bool(squeeze_mask.iloc[-1])
        squeeze_ago = bars_since(squeeze_mask, 20)
        up = bars_since(close > bands["upper"], window)
        down = bars_since(close < bands["lower"], window)

        if squeeze_now:
            fb.event(
                f"Squeeze: bandwidth is in the lowest {pct:g}% of the last "
                f"{len(recent_width)} candles"
            )
        elif squeeze_ago is not None:
            fb.event(f"Squeeze released: bandwidth was compressed {ago(squeeze_ago)}")
        else:
            fb.event("No squeeze in the last 20 candles")
        if up is not None:
            fb.event(f"Close above the upper band {ago(up)}")
        if down is not None:
            fb.event(f"Close below the lower band {ago(down)}")
        fb.event(f"Bandwidth percentile {width_rank:.0f} (lower = more compressed)")

        c = float(close.iloc[-1])
        after_squeeze = squeeze_ago is not None
        if up is not None and (down is None or up < down) and after_squeeze and c > middle:
            fb.signal("BUY", "Upside breakout from a volatility squeeze")
        elif down is not None and (up is None or down < up) and after_squeeze and c < middle:
            fb.signal("SELL", "Downside breakout from a volatility squeeze")
        elif squeeze_now:
            fb.signal("SIDEWAYS", "Squeeze in progress: volatility compressed, awaiting breakout")
        elif up is not None or down is not None:
            fb.signal("SIDEWAYS", "Band break without a prior squeeze (lower-quality setup)")
        else:
            fb.signal("SIDEWAYS", "No squeeze breakout")
