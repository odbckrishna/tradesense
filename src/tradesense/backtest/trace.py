"""One strategy's backtest laid out on its candles: entries, their outcomes, accuracy.

Uses the backtest's scoring exactly (:mod:`tradesense.backtest.scoring`), so ``stats``
matches the Backtest leaderboard for the same candles. An *entry* is the first candle of
a run of the same BUY / SELL signal; it is judged like any signal candle, ``horizon``
candles later, and stays ``pending`` when that candle has not closed yet.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tradesense.backtest.models import HitStats, StrategyOrigin
from tradesense.backtest.scoring import Threshold, score_signals, threshold_at


class TraceCandle(BaseModel):
    time: int  # open time, epoch seconds (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float


class TraceEntry(BaseModel):
    time: int
    signal: Literal["BUY", "SELL"]
    price: float  # close of the entry candle
    judged_time: int | None = None
    judged_price: float | None = None
    move_pips: float | None = None  # in the signalled direction
    hit: bool | None = None  # None: too recent to judge
    candles: int = 1  # how long the signal lasted


class EntryStats(BaseModel):
    entries: int = 0
    hits: int = 0
    hit_rate: float | None = None
    buy_entries: int = 0
    buy_hits: int = 0
    buy_hit_rate: float | None = None
    sell_entries: int = 0
    sell_hits: int = 0
    sell_hit_rate: float | None = None
    pending: int = 0


class TraceData(BaseModel):
    """The computed part of a trace."""

    start: datetime | None = None  # first scored candle
    end: datetime | None = None  # last scored candle
    candles: list[TraceCandle] = Field(default_factory=list)
    signals: list[int | None] = Field(default_factory=list)  # per candle; None = not ready
    entries: list[TraceEntry] = Field(default_factory=list)
    stats: HitStats = Field(default_factory=HitStats)  # every BUY / SELL candle
    entry_stats: EntryStats = Field(default_factory=EntryStats)


class StrategyTrace(TraceData):
    instrument: str
    timeframe: str
    strategy: str
    display_name: str
    origin: StrategyOrigin
    category: str
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime
    duration_ms: int
    horizon_bars: int
    pip_size: float
    spread_pips: float | None = None
    notes: list[str] = Field(default_factory=list)


def _rate(hits: int, n: int) -> float | None:
    return round(hits / n, 4) if n else None


def _epoch(ts: Any) -> int:
    return int(pd.Timestamp(ts).timestamp())


def build_trace(
    *,
    candles: pd.DataFrame,
    signals: np.ndarray,
    moves: np.ndarray,
    start: int,
    end: int,
    horizon: int,
    threshold: Threshold,
    pip: float,
) -> TraceData:
    """Candles from ``start`` to the last one, with entries judged over ``[start, end)``."""
    n = len(candles)
    index = candles.index
    close = candles["close"].to_numpy(dtype=float)
    scored = np.zeros(n, dtype=bool)
    scored[start:end] = True

    entries: list[TraceEntry] = []
    for t in range(start, n):
        code = signals[t]
        if np.isnan(code) or code == 0 or (t > start and signals[t - 1] == code):
            continue
        run = 1
        while t + run < n and signals[t + run] == code:
            run += 1
        direction = 1.0 if code > 0 else -1.0
        entry = TraceEntry(
            time=_epoch(index[t]),
            signal="BUY" if code > 0 else "SELL",
            price=float(close[t]),
            candles=run,
        )
        if t < end and not np.isnan(moves[t]):
            signed = float(moves[t]) * direction
            entry.judged_time = _epoch(index[t + horizon])
            entry.judged_price = float(close[t + horizon])
            entry.move_pips = round(signed / pip, 2) if pip > 0 else None
            entry.hit = signed > threshold_at(threshold, t)
        entries.append(entry)

    judged = [e for e in entries if e.hit is not None]
    buys = [e for e in judged if e.signal == "BUY"]
    sells = [e for e in judged if e.signal == "SELL"]
    hits, buy_hits = sum(e.hit is True for e in judged), sum(e.hit is True for e in buys)
    view = candles.iloc[start:]
    return TraceData(
        start=index[start].to_pydatetime() if start < n else None,
        end=index[end - 1].to_pydatetime() if 0 < end <= n else None,
        candles=[
            TraceCandle(
                time=_epoch(ts),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for ts, row in zip(view.index, view.itertuples(index=False), strict=True)
        ],
        signals=[None if np.isnan(v) else int(v) for v in signals[start:]],
        entries=entries,
        stats=score_signals(signals, moves, threshold=threshold, pip=pip, select=scored),
        entry_stats=EntryStats(
            entries=len(judged),
            hits=hits,
            hit_rate=_rate(hits, len(judged)),
            buy_entries=len(buys),
            buy_hits=buy_hits,
            buy_hit_rate=_rate(buy_hits, len(buys)),
            sell_entries=len(sells),
            sell_hits=hits - buy_hits,
            sell_hit_rate=_rate(hits - buy_hits, len(sells)),
            pending=len(entries) - len(judged),
        ),
    )
