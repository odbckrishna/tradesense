"""VWAP Bounce (levels)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradesense.indicators.technical import ago, atr, bars_since, last_valid, vwap
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, require
from tradesense.strategies.registry import register_strategy


def _streak(mask: pd.Series) -> int:
    count = 0
    for value in reversed(mask.to_list()):
        if not value:
            break
        count += 1
    return count


@register_strategy
class VwapBounce(IndicatorStrategy):
    name = "vwap_bounce"
    display_name = "VWAP Bounce"
    description = (
        "Session VWAP as dynamic support/resistance: a touch of VWAP followed by a close back "
        "on the trend side (bullish or bearish candle) signals a bounce or rejection."
    )
    category = "levels"
    min_candles = 30
    default_params = {"touch_lookback": 5, "atr_period": 14, "max_extension_atr": 1.5}  # noqa: RUF012

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        lookback = int(params["touch_lookback"])
        max_ext = float(params["max_extension_atr"])
        line = vwap(df)
        v, a = require(vwap=last_valid(line), atr=last_valid(atr(df, int(params["atr_period"]))))
        close, open_ = df["close"], df["open"]
        c, o = float(close.iloc[-1]), float(open_.iloc[-1])
        distance_atr = (c - v) / a
        session_start = df.index[-1].floor("D")
        session_candles = int((df.index >= session_start).sum())

        fb.series("close", close)
        fb.series("vwap", line)
        fb.value("distance_atr", distance_atr)
        fb.value("atr", a)
        fb.value("session_candles", session_candles)

        touched = (df["low"] <= line) & (df["high"] >= line)
        touch = bars_since(touched, lookback)
        above = _streak(close > line)
        below = _streak(close < line)
        fb.event(f"Session VWAP built from {session_candles} candles since 00:00 UTC")
        if touch is not None:
            fb.event(f"Price touched VWAP {ago(touch)}")
        if above:
            fb.event(f"Closed above VWAP for {above} consecutive candles")
        if below:
            fb.event(f"Closed below VWAP for {below} consecutive candles")
        fb.event(f"Close is {distance_atr:+.2f} ATR from VWAP")

        bullish_candle, bearish_candle = c > o, c < o
        if session_candles < 3:
            fb.signal("SIDEWAYS", "Session just started: VWAP not yet meaningful")
        elif touch is not None and c > v and bullish_candle and distance_atr <= max_ext:
            fb.signal("BUY", "Bounce off VWAP with a bullish close above it")
        elif touch is not None and c < v and bearish_candle and -distance_atr <= max_ext:
            fb.signal("SELL", "Rejection at VWAP with a bearish close below it")
        elif abs(distance_atr) > max_ext:
            fb.signal("SIDEWAYS", "Price is extended from VWAP; no bounce setup")
        else:
            fb.signal("SIDEWAYS", f"No confirmed VWAP bounce in the last {lookback} candles")
