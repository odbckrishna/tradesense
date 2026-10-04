"""ATR Volatility Breakout (volatility)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradesense.indicators.technical import ago, atr, bars_since, last_valid
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, require
from tradesense.strategies.registry import register_strategy


@register_strategy
class AtrBreakout(IndicatorStrategy):
    name = "atr_breakout"
    display_name = "ATR Volatility Breakout"
    description = (
        "A close beyond the prior N-candle range on a candle whose range exceeds a multiple of "
        "ATR (volatility expansion) signals a breakout in that direction."
    )
    category = "volatility"
    min_candles = 45
    default_params = {  # noqa: RUF012
        "atr_period": 14,
        "lookback": 20,
        "multiplier": 1.0,
        "breakout_window": 3,
    }

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["lookback"]) + int(params["atr_period"]) + 10)

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        lookback, window = int(params["lookback"]), int(params["breakout_window"])
        multiplier = float(params["multiplier"])
        atr_line = atr(df, int(params["atr_period"]))
        prior_high = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
        prior_low = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
        range_atr = (df["high"] - df["low"]) / atr_line.shift(1)
        atr_avg = atr_line.rolling(50, min_periods=20).mean()
        _, hi, lo = require(
            atr=last_valid(atr_line),
            prior_high=last_valid(prior_high),
            prior_low=last_valid(prior_low),
        )
        ratio = last_valid(atr_line / atr_avg)

        fb.series("close", df["close"])
        fb.series("atr", atr_line)
        fb.value("atr_ratio", ratio)
        fb.value("prior_high", hi)
        fb.value("prior_low", lo)
        fb.value("range_atr", last_valid(range_atr))

        expansion = range_atr >= multiplier
        bull = bars_since((df["close"] > prior_high) & expansion, window)
        bear = bars_since((df["close"] < prior_low) & expansion, window)
        if bull is not None:
            fb.event(f"Bullish volatility breakout above the {lookback}-candle high {ago(bull)}")
        if bear is not None:
            fb.event(f"Bearish volatility breakout below the {lookback}-candle low {ago(bear)}")
        if ratio is not None:
            state = "expanding" if ratio > 1.1 else "contracting" if ratio < 0.9 else "normal"
            fb.event(f"Volatility {state}: ATR is {ratio:.2f}x its 50-candle average")
        c = float(df["close"].iloc[-1])
        fb.event(
            f"Close sits in the prior range [{lo:.6g}, {hi:.6g}]"
            if lo <= c <= hi
            else f"Close is outside the prior {lookback}-candle range"
        )

        if bull is not None and (bear is None or bull < bear):
            fb.signal("BUY", "Upside range breakout with volatility expansion")
        elif bear is not None:
            fb.signal("SELL", "Downside range breakout with volatility expansion")
        else:
            fb.signal("SIDEWAYS", f"No volatility breakout in the last {window} candles")
