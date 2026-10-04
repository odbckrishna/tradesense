"""Directional hit rate of a signal series.

A signal at candle ``t`` is computed from candles up to ``t`` only. It is judged by the
move ``close[t + horizon] - close[t]``: a BUY (SELL) is a hit when price rose (fell) by
more than ``threshold``, the spread (one value, or one per candle: the spread when that
signal happened), so moves too small to pay for the trade are misses.
Signals are encoded as +1 BUY, -1 SELL, 0 SIDEWAYS and NaN (strategy not ready).
"""

from __future__ import annotations

import math

import numpy as np

from tradesense.backtest.models import HitStats

Z_95 = 1.96

#: The move a hit must beat: one value, or one per candle (aligned with the signals).
type Threshold = float | np.ndarray


def threshold_at(threshold: Threshold, index: int) -> float:
    return float(threshold[index]) if isinstance(threshold, np.ndarray) else threshold


def forward_moves(close: np.ndarray, horizon: int) -> np.ndarray:
    """``close[t + horizon] - close[t]`` per candle; NaN where the future is unknown."""
    moves = np.full(len(close), np.nan)
    if horizon < len(close):
        moves[:-horizon] = close[horizon:] - close[:-horizon]
    return moves


def wilson_lower(hits: int, n: int, z: float = Z_95) -> float:
    """Lower bound of the Wilson score interval: a hit rate that small samples can't game."""
    if n <= 0:
        return 0.0
    p = hits / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denominator)


def _rate(hits: int, n: int) -> float | None:
    return round(hits / n, 4) if n else None


def score_signals(
    signals: np.ndarray,
    moves: np.ndarray,
    *,
    threshold: Threshold,
    pip: float,
    select: np.ndarray | None = None,
) -> HitStats:
    """Hit statistics over the candles in ``select`` (a boolean mask) that have both a
    signal and a known outcome."""
    valid = ~np.isnan(signals) & ~np.isnan(moves)
    if select is not None:
        valid &= select
    sig, move = signals[valid], moves[valid]
    buy, sell = sig > 0, sig < 0
    directional = buy | sell
    direction = np.sign(sig[directional])
    signed = move[directional] * direction  # positive = price went the signalled way
    if isinstance(threshold, np.ndarray):
        hit = signed > threshold[valid][directional]
    else:
        hit = signed > threshold
    buys, sells = int(buy.sum()), int(sell.sum())
    hits = int(hit.sum())
    buy_hits = int((hit & (direction > 0)).sum())

    # A new episode starts where the signal changes to BUY or SELL (on consecutive candles).
    full = np.where(valid, signals, np.nan)
    previous = np.concatenate([[np.nan], full[:-1]])
    starts = valid & (full != 0) & (full != previous)

    n = buys + sells
    return HitStats(
        signals=n,
        buys=buys,
        sells=sells,
        sideways=int((sig == 0).sum()),
        hits=hits,
        buy_hits=buy_hits,
        sell_hits=hits - buy_hits,
        episodes=int(starts.sum()),
        hit_rate=_rate(hits, n),
        buy_hit_rate=_rate(buy_hits, buys),
        sell_hit_rate=_rate(hits - buy_hits, sells),
        score=round(wilson_lower(hits, n), 4),
        avg_move_pips=round(float(signed.mean()) / pip, 2) if n and pip > 0 else None,
    )


def period_masks(length: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Masks for the scored period ``[start, end)`` and its first (train) and second
    (validation) halves, split in time."""
    whole = np.zeros(length, dtype=bool)
    whole[start:end] = True
    middle = start + (end - start) // 2
    train = np.zeros(length, dtype=bool)
    train[start:middle] = True
    return whole, train, whole & ~train
