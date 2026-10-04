"""Pivot Point Support / Resistance (levels)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradesense.indicators.technical import (
    ago,
    atr,
    bars_since,
    crossed_above,
    crossed_below,
    last_valid,
    pivot_levels,
)
from tradesense.strategies.base import (
    FeatureBuilder,
    IndicatorStrategy,
    InsufficientDataError,
    StrategyError,
    require,
)
from tradesense.strategies.registry import register_strategy

_ORDER = ("s3", "s2", "s1", "pivot", "r1", "r2", "r3")


@register_strategy
class PivotPoints(IndicatorStrategy):
    name = "pivot_points"
    display_name = "Pivot Points S/R"
    description = (
        "Classic floor pivots from the previous period: holding above the pivot with room to "
        "resistance is bullish, below with room to support is bearish; at a level is SIDEWAYS."
    )
    category = "levels"
    min_candles = 30
    default_params = {"period": "4h", "proximity_atr": 0.3, "atr_period": 14}  # noqa: RUF012

    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        period = str(params["period"])
        proximity = float(params["proximity_atr"])
        try:
            levels = pivot_levels(df, period)
        except ValueError as exc:
            raise StrategyError(str(exc)) from None
        if levels is None:
            raise InsufficientDataError(
                f"Pivot period '{period}' needs one complete previous period in the data "
                "(use a shorter 'period' or more candles)."
            )
        (a,) = require(atr=last_valid(atr(df, int(params["atr_period"]))))
        c = float(df["close"].iloc[-1])
        fb.series("close", df["close"])
        for name in _ORDER:
            fb.value(name, levels[name])
        fb.value("atr", a)
        fb.value("period", period)

        supports = sorted(((v, k) for k, v in levels.items() if v < c), reverse=True)
        resistances = sorted((v, k) for k, v in levels.items() if v > c)
        if supports:
            value, name = supports[0]
            fb.value("nearest_support", value)
            fb.value("support_distance_atr", (c - value) / a)
            fb.event(f"Nearest support {name.upper()} at {value:.6g} ({(c - value) / a:.2f} ATR)")
        if resistances:
            value, name = resistances[0]
            fb.value("nearest_resistance", value)
            fb.value("resistance_distance_atr", (value - c) / a)
            fb.event(
                f"Nearest resistance {name.upper()} at {value:.6g} ({(value - c) / a:.2f} ATR)"
            )
        pivot = levels["pivot"]
        fb.event(f"Price is {'above' if c > pivot else 'below'} the {period} pivot ({pivot:.6g})")
        for name in ("r1", "r2", "s1", "s2"):
            up = bars_since(crossed_above(df["close"], levels[name]), 10)
            down = bars_since(crossed_below(df["close"], levels[name]), 10)
            if up is not None:
                fb.event(f"Close broke above {name.upper()} {ago(up)}")
            if down is not None:
                fb.event(f"Close broke below {name.upper()} {ago(down)}")

        near_support = bool(supports) and (c - supports[0][0]) / a <= proximity
        near_resistance = bool(resistances) and (resistances[0][0] - c) / a <= proximity
        if abs(c - pivot) / a <= proximity:
            fb.signal("SIDEWAYS", "Price is sitting on the pivot")
        elif c > pivot:
            if near_resistance:
                fb.signal("SIDEWAYS", f"Bullish bias but testing resistance {resistances[0][1]}")
            else:
                fb.signal("BUY", "Holding above the pivot with room to the next resistance")
        elif near_support:
            fb.signal("SIDEWAYS", f"Bearish bias but testing support {supports[0][1]}")
        else:
            fb.signal("SELL", "Holding below the pivot with room to the next support")
