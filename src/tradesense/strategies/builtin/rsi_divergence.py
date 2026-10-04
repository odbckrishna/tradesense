"""RSI Divergence (momentum)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradesense.indicators.technical import last_valid, rsi, swing_points
from tradesense.strategies.base import FeatureBuilder, IndicatorStrategy, require
from tradesense.strategies.registry import register_strategy


def _last_two(swings: pd.Series, line: pd.Series, n: int) -> list[tuple[int, float, float]]:
    """(candles_ago, price, rsi) for the two most recent swings."""
    points = []
    for ts, price in swings.dropna().iloc[-2:].items():
        position = int(swings.index.get_loc(ts))
        points.append((n - 1 - position, float(price), float(line.loc[ts])))
    return points


@register_strategy
class RsiDivergence(IndicatorStrategy):
    name = "rsi_divergence"
    display_name = "RSI Divergence"
    description = (
        "Price making a lower low while RSI makes a higher low is bullish divergence; a higher "
        "high with a lower RSI high is bearish divergence (momentum not confirming price)."
    )
    category = "momentum"
    min_candles = 60
    default_params = {  # noqa: RUF012
        "period": 14,
        "swing_left": 3,
        "swing_right": 3,
        "lookback": 60,
        "max_age": 12,
    }

    def required_candles(self, params: Mapping[str, Any]) -> int:
        return max(self.min_candles, int(params["lookback"]) + int(params["period"]) + 5)

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        lookback, max_age = int(params["lookback"]), int(params["max_age"])
        line = rsi(df["close"], int(params["period"]))
        (r,) = require(rsi=last_valid(line))
        window = df.iloc[-lookback:]
        swings = swing_points(window, int(params["swing_left"]), int(params["swing_right"]))
        n = len(window)
        lows = _last_two(swings["pivot_low"], line, n)
        highs = _last_two(swings["pivot_high"], line, n)

        fb.series("rsi", line)
        fb.series("close", df["close"])
        bullish = bearish = None
        if len(lows) == 2:
            (_, p1, r1), (age2, p2, r2) = lows
            fb.value("swing_low_prev", p1)
            fb.value("swing_low_last", p2)
            fb.value("rsi_at_low_prev", r1)
            fb.value("rsi_at_low_last", r2)
            if p2 < p1 and r2 > r1:
                bullish = age2
                fb.event(
                    f"Bullish divergence: lower low in price ({p2:.6g} < {p1:.6g}) with a "
                    f"higher RSI low ({r2:.1f} > {r1:.1f}); latest swing {age2} candles ago"
                )
        if len(highs) == 2:
            (_, p1, r1), (age2, p2, r2) = highs
            fb.value("swing_high_prev", p1)
            fb.value("swing_high_last", p2)
            fb.value("rsi_at_high_prev", r1)
            fb.value("rsi_at_high_last", r2)
            if p2 > p1 and r2 < r1:
                bearish = age2
                fb.event(
                    f"Bearish divergence: higher high in price ({p2:.6g} > {p1:.6g}) with a "
                    f"lower RSI high ({r2:.1f} < {r1:.1f}); latest swing {age2} candles ago"
                )
        if bullish is None and bearish is None:
            fb.event(f"No RSI divergence between the last two swings (last {lookback} candles)")
        fb.event(f"RSI is {r:.1f}")

        fresh_bull = bullish is not None and bullish <= max_age
        fresh_bear = bearish is not None and bearish <= max_age
        if fresh_bull and (not fresh_bear or (bearish or 0) > (bullish or 0)):
            fb.signal("BUY", "Recent bullish RSI divergence: selling momentum is fading")
        elif fresh_bear:
            fb.signal("SELL", "Recent bearish RSI divergence: buying momentum is fading")
        elif bullish is not None or bearish is not None:
            fb.signal("SIDEWAYS", f"Divergence is older than {max_age} candles")
        else:
            fb.signal("SIDEWAYS", "No divergence")
