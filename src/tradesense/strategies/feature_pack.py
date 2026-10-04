"""Standard indicator bundle shared by all reference (document / image) strategies.

Computed once per timeframe per run: EMA 20/50/200, RSI 14, MACD, Bollinger Bands,
ATR 14, ADX, recent swing highs/lows and volume change.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from tradesense.indicators import technical as ti
from tradesense.strategies.base import round_value, series_tail


def compute_feature_pack(df: pd.DataFrame, history_length: int = 10) -> dict[str, Any]:
    """Return ``{"latest", "history", "swings", "volume", "summary"}`` for ``df``."""
    close = df["close"]
    ema20, ema50, ema200 = ti.ema(close, 20), ti.ema(close, 50), ti.ema(close, 200)
    rsi14 = ti.rsi(close, 14)
    macd = ti.macd(close)
    bands = ti.bbands(close, 20, 2.0)
    atr14 = ti.atr(df, 14)
    adx = ti.adx(df, 14)
    volume = ti.volume_profile(df, 20)

    def last(series: pd.Series) -> float | None:
        return round_value(ti.last_valid(series))

    latest: dict[str, float | None] = {
        "close": round_value(float(close.iloc[-1])),
        "ema20": last(ema20),
        "ema50": last(ema50),
        "ema200": last(ema200),
        "rsi14": last(rsi14),
        "macd": last(macd["macd"]),
        "macd_signal": last(macd["signal"]),
        "macd_hist": last(macd["hist"]),
        "bb_upper": last(bands["upper"]),
        "bb_middle": last(bands["middle"]),
        "bb_lower": last(bands["lower"]),
        "bb_percent_b": last(bands["percent_b"]),
        "bb_width_pct": last(bands["width"]),
        "atr14": last(atr14),
        "adx14": last(adx["adx"]),
        "plus_di": last(adx["plus_di"]),
        "minus_di": last(adx["minus_di"]),
    }
    history = {
        "close": series_tail(close, history_length),
        "ema20": series_tail(ema20, history_length),
        "rsi14": series_tail(rsi14, history_length),
        "macd_hist": series_tail(macd["hist"], history_length),
    }
    recent = df.iloc[-5:]["volume"].mean()
    prior = df.iloc[-25:-5]["volume"].mean() if len(df) > 25 else None
    volume_info = {
        "last": round_value(float(df["volume"].iloc[-1])),
        "sma20": last(volume["sma"]),
        "ratio_to_sma20": last(volume["ratio"]),
        "change_5_vs_prior_20_pct": round_value((recent / prior - 1) * 100) if prior else None,
    }
    return {
        "candles": len(df),
        "latest": latest,
        "history": history,
        "swings": ti.recent_swings(df, count=3),
        "volume": volume_info,
        "summary": list(_describe(latest, macd["hist"], volume_info)),
    }


def _describe(
    latest: dict[str, float | None], hist: pd.Series, volume: dict[str, Any]
) -> Iterable[str]:
    """Plain-language facts derived from the pack (become prompt 'events')."""
    close = latest["close"]
    emas = {k: latest[k] for k in ("ema20", "ema50", "ema200") if latest[k] is not None}
    if close is not None and emas:
        above = [k.upper() for k, v in emas.items() if v is not None and close > v]
        below = [k.upper() for k, v in emas.items() if v is not None and close < v]
        if above:
            yield f"Close is above {', '.join(above)}"
        if below:
            yield f"Close is below {', '.join(below)}"
    e20, e50, e200 = latest["ema20"], latest["ema50"], latest["ema200"]
    if e20 is not None and e50 is not None:
        if e200 is not None and e20 > e50 > e200:
            yield "EMAs stacked bullish (20 > 50 > 200)"
        elif e200 is not None and e20 < e50 < e200:
            yield "EMAs stacked bearish (20 < 50 < 200)"
        else:
            yield f"EMA20 is {'above' if e20 > e50 else 'below'} EMA50"
    if latest["ema200"] is None:
        yield "EMA200 not available (fewer than 200 candles)"
    rsi = latest["rsi14"]
    if rsi is not None:
        zone = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
        yield f"RSI14 at {rsi:.1f} ({zone})"
    valid_hist = hist.dropna()
    if len(valid_hist) >= 2:
        rising = valid_hist.iloc[-1] > valid_hist.iloc[-2]
        sign = "positive" if valid_hist.iloc[-1] > 0 else "negative"
        yield f"MACD histogram {sign} and {'rising' if rising else 'falling'}"
    adx = latest["adx14"]
    if adx is not None:
        yield f"ADX14 at {adx:.1f} ({'trending' if adx >= 25 else 'weak trend / range'})"
    pct_b = latest["bb_percent_b"]
    if pct_b is not None:
        where = (
            "near the upper band"
            if pct_b > 0.8
            else "near the lower band"
            if pct_b < 0.2
            else "mid-band"
        )
        yield f"Bollinger %B {pct_b:.2f} ({where})"
    ratio = volume.get("ratio_to_sma20")
    if ratio is not None:
        yield f"Volume is {ratio:.2f}x its 20-candle average"


def compute_required_indicators(
    df: pd.DataFrame, names: Iterable[str], history_length: int = 10
) -> dict[str, dict[str, Any]]:
    """Latest value (all outputs) and recent history (primary output) per indicator."""
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        kind = ti.INDICATORS.get(name)
        if kind is None:
            continue
        frame = kind.run(df, {})
        out[name] = {
            "params": dict(kind.defaults),
            "latest": {o: round_value(ti.last_valid(frame[o])) for o in kind.outputs},
            "history": series_tail(frame[kind.primary], history_length),
        }
    return out
