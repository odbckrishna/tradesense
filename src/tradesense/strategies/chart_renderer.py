"""Renders live candles to PNG with mplfinance (consistent style for vision models).

matplotlib is not thread-safe, so rendering is serialised with a lock and run in a
worker thread by :func:`render_chart_async`.
"""

from __future__ import annotations

import asyncio
import io
import threading
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from tradesense.indicators import technical as ti

_LOCK = threading.Lock()

STYLE = mpf.make_mpf_style(
    base_mpf_style="yahoo",
    marketcolors=mpf.make_marketcolors(
        up="#16a34a", down="#dc2626", edge="inherit", wick="inherit", volume="in"
    ),
    gridstyle=":",
    gridcolor="#d4d4d8",
    facecolor="#ffffff",
    figcolor="#ffffff",
    rc={"font.size": 9, "axes.titleweight": "bold"},
)


def _usable(series: pd.Series) -> bool:
    return bool(series.notna().any())


def render_chart(
    df: pd.DataFrame,
    *,
    title: str = "",
    indicators: Sequence[str] = (),
    last_n: int = 100,
    size_px: tuple[int, int] = (1200, 720),
) -> bytes:
    """PNG of the last ``last_n`` candles with overlays for the requested indicators.

    EMA 20/50 are always drawn; ``bbands``, ``vwap``, ``supertrend``, ``sma`` and
    ``swing_points`` add overlays; ``rsi`` / ``macd`` add lower panels; volume is
    always shown.
    """
    if len(df) < 2:
        raise ValueError("need at least two candles to render a chart")
    wanted = set(indicators)
    window = slice(-last_n, None)
    data = df.iloc[window].rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    close = df["close"]
    plots: list[Any] = []

    def overlay(series: pd.Series, **kwargs: Any) -> None:
        part = series.iloc[window]
        if _usable(part):
            plots.append(mpf.make_addplot(part, **kwargs))

    overlay(ti.ema(close, 20), color="#2563eb", width=1.0)
    overlay(ti.ema(close, 50), color="#f59e0b", width=1.0)
    if "sma" in wanted:
        overlay(ti.sma(close, 20), color="#7c3aed", width=0.9)
    if "bbands" in wanted:
        bands = ti.bbands(close)
        overlay(bands["upper"], color="#64748b", linestyle="--", width=0.8)
        overlay(bands["lower"], color="#64748b", linestyle="--", width=0.8)
    if "vwap" in wanted:
        overlay(ti.vwap(df), color="#db2777", width=1.0)
    if "supertrend" in wanted:
        overlay(ti.supertrend(df)["value"], color="#0f766e", width=1.0)
    if "swing_points" in wanted:
        swings = ti.swing_points(df)
        overlay(
            swings["pivot_high"] * 1.0002,
            type="scatter",
            marker="v",
            markersize=30,
            color="#dc2626",
        )
        overlay(
            swings["pivot_low"] * 0.9998, type="scatter", marker="^", markersize=30, color="#16a34a"
        )

    panel = 2  # next free panel: 0 = price, 1 = volume
    if "rsi" in wanted:
        overlay(ti.rsi(close), panel=panel, color="#7c3aed", ylabel="RSI", ylim=(0, 100))
        panel += 1
    if "macd" in wanted:
        lines = ti.macd(close)
        overlay(lines["macd"], panel=panel, color="#2563eb", ylabel="MACD")
        overlay(lines["signal"], panel=panel, color="#f59e0b")
        overlay(lines["hist"], panel=panel, type="bar", color="#94a3b8")
        panel += 1

    width, height = size_px
    ratios = (5.0, 1.2, *([1.6] * (panel - 2)))
    with _LOCK:
        fig, axes = mpf.plot(
            data,
            type="candle",
            style=STYLE,
            addplot=plots or None,
            volume=True,
            returnfig=True,
            figsize=(width / 100, height / 100),
            axtitle=title,
            panel_ratios=ratios,
            tight_layout=True,
            scale_padding={"left": 0.4, "right": 2.0, "top": 1.4, "bottom": 0.8},
            warn_too_much_data=100_000,
        )
        # Plain price labels (no "+1.08e0" offset notation that confuses vision models).
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axes[0].yaxis.set_major_formatter(formatter)
        buffer = io.BytesIO()
        try:
            fig.savefig(buffer, format="png", dpi=100)
        finally:
            plt.close(fig)
    return buffer.getvalue()


async def render_chart_async(df: pd.DataFrame, **kwargs: Any) -> bytes:
    return await asyncio.to_thread(render_chart, df, **kwargs)
