"""RSI Extremes / Mean Reversion."""

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
    rsi,
)
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, StrategyError, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class RsiExtremes(IndicatorStrategy):
    name = "rsi_extremes"
    display_name = "RSI Extremes"
    description = (
        "Mean reversion: RSI leaving the oversold zone suggests a bounce (BUY), leaving the "
        "overbought zone suggests a pullback (SELL)."
    )
    category = "mean_reversion"
    min_candles = 30
    default_params = {"period": 14, "overbought": 70.0, "oversold": 30.0, "lookback": 5}  # noqa: RUF012

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["period"]) * 3 + 1)

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        period, lookback = int(params["period"]), int(params["lookback"])
        overbought, oversold = float(params["overbought"]), float(params["oversold"])
        if not 0 < oversold < overbought < 100:
            raise StrategyError("require 0 < oversold < overbought < 100")
        line = rsi(df["close"], period)
        r, r_prev = require(rsi=last_valid(line), rsi_previous=last_valid(line.iloc[:-1]))

        fb.series("rsi", line)
        fb.series("close", df["close"])
        fb.value("rsi_change", r - r_prev)
        fb.value("rsi_min_recent", float(line.iloc[-lookback:].min()))
        fb.value("rsi_max_recent", float(line.iloc[-lookback:].max()))

        exit_oversold = bars_since(crossed_above(line, oversold), lookback)
        exit_overbought = bars_since(crossed_below(line, overbought), lookback)
        if r <= oversold:
            fb.event(f"RSI is oversold at {r:.1f} (below {oversold:g})")
        elif r >= overbought:
            fb.event(f"RSI is overbought at {r:.1f} (above {overbought:g})")
        else:
            fb.event(f"RSI is neutral at {r:.1f}")
        if exit_oversold is not None:
            fb.event(f"RSI crossed back above {oversold:g} (oversold exit) {ago(exit_oversold)}")
        if exit_overbought is not None:
            fb.event(
                f"RSI crossed back below {overbought:g} (overbought exit) {ago(exit_overbought)}"
            )
        rising = r > r_prev
        fb.event("RSI is rising" if rising else "RSI is falling")

        if exit_oversold is not None and r < 50:
            fb.signal("BUY", "RSI is exiting the oversold zone")
        elif exit_overbought is not None and r > 50:
            fb.signal("SELL", "RSI is exiting the overbought zone")
        elif r <= oversold:
            if rising:
                fb.signal("BUY", "RSI oversold and turning up")
            else:
                fb.signal("SIDEWAYS", "RSI oversold but still falling: wait for a turn")
        elif r >= overbought:
            if not rising:
                fb.signal("SELL", "RSI overbought and turning down")
            else:
                fb.signal("SIDEWAYS", "RSI overbought but still rising: wait for a turn")
        else:
            fb.signal("SIDEWAYS", "RSI is in the neutral zone")
