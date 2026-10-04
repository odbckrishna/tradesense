"""Shared technical-indicator helpers.

Every function takes time-indexed pandas objects and returns Series/DataFrames
aligned to the input index. Values that cannot be computed yet (warm-up) are NaN;
too-short inputs return all-NaN output instead of raising.

Standard indicators use the ``ta`` library; Supertrend, session VWAP, Ichimoku,
swing points and pivot levels are implemented here.

:data:`INDICATORS` is the registry of indicator *types* usable from YAML rule
strategies and reference-strategy cards (``ema``, ``rsi``, ``macd`` ...).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import ta

# --------------------------------------------------------------------------- utils


def _nan_like(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def crossed_above(a: pd.Series, b: pd.Series | float) -> pd.Series:
    """True on candles where ``a`` crosses from <= b to > b."""
    b_series = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    result = (a > b_series) & (a.shift(1) <= b_series.shift(1))
    return result.fillna(False).astype(bool)


def crossed_below(a: pd.Series, b: pd.Series | float) -> pd.Series:
    """True on candles where ``a`` crosses from >= b to < b."""
    b_series = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    result = (a < b_series) & (a.shift(1) >= b_series.shift(1))
    return result.fillna(False).astype(bool)


def bars_since(mask: pd.Series, lookback: int | None = None) -> int | None:
    """Candles since ``mask`` was last True (0 = latest candle), or None."""
    values = mask.to_numpy(dtype=bool)
    if lookback is not None:
        values = values[-lookback:]
    hits = np.flatnonzero(values)
    if hits.size == 0:
        return None
    return int(len(values) - 1 - hits[-1])


def last_valid(series: pd.Series) -> float | None:
    """Latest non-NaN value as a float."""
    valid = series.dropna()
    if valid.empty:
        return None
    value = float(valid.iloc[-1])
    return value if math.isfinite(value) else None


def ago(n: int) -> str:
    """Human phrase for a candle offset used in strategy events."""
    if n == 0:
        return "on the latest candle"
    return f"{n} candle{'s' if n != 1 else ''} ago"


def infer_bar_seconds(index: pd.DatetimeIndex) -> float | None:
    """Median spacing of the index in seconds (unit-safe: pandas 3 defaults to us)."""
    if len(index) < 2:
        return None
    return float(index.to_series().diff().dt.total_seconds().median())


# --------------------------------------------------------------------------- moving averages


def ema(close: pd.Series, period: int) -> pd.Series:
    if len(close) < period:
        return _nan_like(close.index)
    return ta.trend.EMAIndicator(close, window=period).ema_indicator()


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


# --------------------------------------------------------------------------- oscillators


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if len(close) <= period:
        return _nan_like(close.index)
    return ta.momentum.RSIIndicator(close, window=period).rsi()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if len(close) < slow + signal:
        nan = _nan_like(close.index)
        return pd.DataFrame({"macd": nan, "signal": nan, "hist": nan})
    ind = ta.trend.MACD(close, window_slow=slow, window_fast=fast, window_sign=signal)
    return pd.DataFrame({"macd": ind.macd(), "signal": ind.macd_signal(), "hist": ind.macd_diff()})


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    """Slow stochastic: %K smoothed over ``smooth`` candles, %D = SMA(%K, d)."""
    lowest = df["low"].rolling(k, min_periods=k).min()
    highest = df["high"].rolling(k, min_periods=k).max()
    span = (highest - lowest).replace(0, np.nan)
    raw_k = 100 * (df["close"] - lowest) / span
    k_line = raw_k.rolling(smooth, min_periods=smooth).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"k": k_line, "d": d_line})


# --------------------------------------------------------------------------- volatility


def bbands(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    if len(close) < period:
        nan = _nan_like(close.index)
        return pd.DataFrame(
            {"middle": nan, "upper": nan, "lower": nan, "width": nan, "percent_b": nan}
        )
    ind = ta.volatility.BollingerBands(close, window=period, window_dev=std)
    return pd.DataFrame(
        {
            "middle": ind.bollinger_mavg(),
            "upper": ind.bollinger_hband(),
            "lower": ind.bollinger_lband(),
            "width": ind.bollinger_wband(),  # (upper - lower) / middle * 100
            "percent_b": ind.bollinger_pband(),
        }
    )


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR, identical to ``ta.volatility.AverageTrueRange`` but vectorised (ta's
    per-candle Python loop was most of every strategy's run time)."""
    if len(df) <= period:
        return _nan_like(df.index)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    prev_close = np.concatenate([[np.nan], df["close"].to_numpy(dtype=float)[:-1]])
    # fmax ignores NaN, so the first true range is high - low (as in ta)
    true_range = np.fmax(high - low, np.fmax(np.abs(high - prev_close), np.abs(low - prev_close)))
    # Seed with the mean of the first `period` true ranges, then Wilder-smooth:
    # atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period, i.e. an EWM with alpha=1/period.
    seeded = np.full(len(true_range), np.nan)
    seeded[period - 1] = true_range[:period].mean()
    seeded[period:] = true_range[period:]
    out = pd.Series(seeded, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    return out.where(out > 0)  # warm-up (and flat data) are not readings


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend line and direction (+1 up-trend, -1 down-trend)."""
    n = len(df)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    atr_values = atr(df, period).to_numpy()
    if n <= period:
        return pd.DataFrame({"value": st, "direction": direction}, index=df.index)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr_values
    lower = hl2 - multiplier * atr_values
    final_upper = upper.copy()
    final_lower = lower.copy()
    start = int(np.argmax(~np.isnan(atr_values)))
    if np.isnan(atr_values[start]):
        return pd.DataFrame({"value": st, "direction": direction}, index=df.index)
    direction[start] = 1.0
    st[start] = final_lower[start]
    for i in range(start + 1, n):
        if upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i - 1]
        if lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i - 1]
        if direction[i - 1] > 0:
            direction[i] = -1.0 if close[i] < final_lower[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] > final_upper[i] else -1.0
        st[i] = final_lower[i] if direction[i] > 0 else final_upper[i]
    return pd.DataFrame({"value": st, "direction": direction}, index=df.index)


# --------------------------------------------------------------------------- trend


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX with +DI / -DI using Wilder smoothing.

    Implemented natively: ``ta.trend.ADXIndicator`` (0.11) leaves its final smoothed
    value at zero and reads true range one candle ahead, corrupting the latest reading.
    """
    if len(df) < 2 * period:
        nan = _nan_like(df.index)
        return pd.DataFrame({"adx": nan, "plus_di": nan, "minus_di": nan})
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    def wilder(series: pd.Series) -> pd.Series:
        return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    smoothed_tr = wilder(true_range).replace(0, np.nan)
    plus_di = 100 * wilder(plus_dm) / smoothed_tr
    minus_di = 100 * wilder(minus_dm) / smoothed_tr
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return pd.DataFrame(
        {"adx": wilder(dx), "plus_di": plus_di, "minus_di": minus_di}, index=df.index
    )


def ichimoku(
    df: pd.DataFrame,
    conversion: int = 9,
    base: int = 26,
    span_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Ichimoku lines. ``senkou_a``/``senkou_b`` are the cloud *at* each candle
    (i.e. already displaced forward), so the latest row is the current cloud."""
    high, low = df["high"], df["low"]

    def mid(window: int) -> pd.Series:
        return (
            high.rolling(window, min_periods=window).max()
            + low.rolling(window, min_periods=window).min()
        ) / 2

    tenkan = mid(conversion)
    kijun = mid(base)
    raw_a = (tenkan + kijun) / 2
    raw_b = mid(span_b)
    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": raw_a.shift(displacement),
            "senkou_b": raw_b.shift(displacement),
            "future_a": raw_a,  # cloud projected ``displacement`` candles ahead
            "future_b": raw_b,
            "chikou_delta": df["close"] - df["close"].shift(displacement),
        }
    )


# --------------------------------------------------------------------------- volume / levels


def vwap(df: pd.DataFrame, anchor: str = "D") -> pd.Series:
    """Session VWAP, reset at each ``anchor`` boundary (UTC days by default).

    OANDA volume is tick volume; if a session has no volume, candles are equally weighted.
    """
    if df.empty:
        return _nan_like(df.index)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].astype("float64")
    session = df.index.floor(anchor)
    session_volume = volume.groupby(session).transform("sum")
    weights = volume.where(session_volume > 0, 1.0)
    cum_pv = (typical * weights).groupby(session).cumsum()
    cum_v = weights.groupby(session).cumsum()
    return (cum_pv / cum_v).astype("float64")


def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Fractal swing highs/lows: ``pivot_high`` / ``pivot_low`` hold the price at the
    swing candle (NaN elsewhere). The last ``right`` candles cannot be confirmed yet."""
    window = left + right + 1
    high, low = df["high"], df["low"]
    # A window ending ``right`` candles after i covers [i - left, i + right].
    roll_max = high.rolling(window, min_periods=window).max().shift(-right)
    roll_min = low.rolling(window, min_periods=window).min().shift(-right)
    pivot_high = high.where(high == roll_max)
    pivot_low = low.where(low == roll_min)
    return pd.DataFrame({"pivot_high": pivot_high, "pivot_low": pivot_low})


def recent_swings(
    df: pd.DataFrame, count: int = 3, left: int = 3, right: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """The ``count`` most recent swing highs and lows with their age in candles."""
    swings = swing_points(df, left, right)
    n = len(df)
    out: dict[str, list[dict[str, Any]]] = {"highs": [], "lows": []}
    for key, column in (("highs", "pivot_high"), ("lows", "pivot_low")):
        series = swings[column].dropna()
        for ts, price in series.iloc[-count:].items():
            position = df.index.get_loc(ts)
            age = n - 1 - int(position) if isinstance(position, int | np.integer) else None
            out[key].append(
                {"time": pd.Timestamp(ts).isoformat(), "price": float(price), "candles_ago": age}
            )
    return out


def pivot_levels(df: pd.DataFrame, period: str = "1D") -> dict[str, float] | None:
    """Classic floor pivots from the previous *complete* ``period`` in ``df``.

    Returns None when the data does not contain a sufficiently complete previous period.
    """
    if df.empty:
        return None
    try:
        period_seconds = pd.Timedelta(period).total_seconds()
        buckets = df.index.floor(period)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid pivot period {period!r}; use e.g. 1h, 4h, 1D") from None
    if period_seconds <= 0:
        raise ValueError(f"Invalid pivot period {period!r}")
    keys = buckets.unique()
    if len(keys) < 2:
        return None
    previous = df[buckets == keys[-2]]
    bar_seconds = infer_bar_seconds(df.index) or period_seconds
    expected = max(1.0, period_seconds / bar_seconds)
    if len(previous) < 0.5 * expected:
        return None
    high = float(previous["high"].max())
    low = float(previous["low"].min())
    close = float(previous["close"].iloc[-1])
    pivot = (high + low + close) / 3
    return {
        "pivot": pivot,
        "r1": 2 * pivot - low,
        "s1": 2 * pivot - high,
        "r2": pivot + (high - low),
        "s2": pivot - (high - low),
        "r3": high + 2 * (pivot - low),
        "s3": low - 2 * (high - pivot),
    }


def volume_profile(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    volume = df["volume"].astype("float64")
    avg = volume.rolling(period, min_periods=period).mean()
    return pd.DataFrame({"value": volume, "sma": avg, "ratio": volume / avg.replace(0, np.nan)})


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class IndicatorType:
    """An indicator usable from YAML rules and strategy cards."""

    name: str
    defaults: Mapping[str, int | float]
    outputs: tuple[str, ...]  # first output is the primary value
    compute: Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]
    warmup: Callable[[Mapping[str, Any]], int]
    description: str

    @property
    def primary(self) -> str:
        return self.outputs[0]

    def resolve_params(self, params: Mapping[str, Any]) -> dict[str, int | float]:
        """Validate and coerce parameters against the defaults."""
        resolved: dict[str, int | float] = dict(self.defaults)
        for key, value in params.items():
            if key not in self.defaults:
                raise ValueError(
                    f"indicator '{self.name}' has no parameter '{key}' "
                    f"(allowed: {', '.join(self.defaults) or 'none'})"
                )
            default = self.defaults[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"indicator '{self.name}' parameter '{key}' must be a number")
            if isinstance(default, int):
                if float(value) != int(value):
                    raise ValueError(f"indicator '{self.name}' parameter '{key}' must be whole")
                value = int(value)
                if not 1 <= value <= 1000:
                    raise ValueError(f"indicator '{self.name}' parameter '{key}' out of range")
            else:
                value = float(value)
                if not 0 < value <= 100:
                    raise ValueError(f"indicator '{self.name}' parameter '{key}' out of range")
            resolved[key] = value
        return resolved

    def run(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        frame = self.compute(df, self.resolve_params(params))
        return frame[list(self.outputs)]


def _i(params: Mapping[str, Any], key: str) -> int:
    return int(params[key])


def _f(params: Mapping[str, Any], key: str) -> float:
    return float(params[key])


def _frame(**series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(series)


INDICATORS: dict[str, IndicatorType] = {
    t.name: t
    for t in (
        IndicatorType(
            "ema",
            {"period": 20},
            ("value",),
            lambda df, p: _frame(value=ema(df["close"], _i(p, "period"))),
            lambda p: _i(p, "period"),
            "Exponential moving average of close",
        ),
        IndicatorType(
            "sma",
            {"period": 20},
            ("value",),
            lambda df, p: _frame(value=sma(df["close"], _i(p, "period"))),
            lambda p: _i(p, "period"),
            "Simple moving average of close",
        ),
        IndicatorType(
            "rsi",
            {"period": 14},
            ("value",),
            lambda df, p: _frame(value=rsi(df["close"], _i(p, "period"))),
            lambda p: _i(p, "period") * 2,
            "Wilder RSI (0-100)",
        ),
        IndicatorType(
            "macd",
            {"fast": 12, "slow": 26, "signal": 9},
            ("macd", "signal", "hist"),
            lambda df, p: macd(df["close"], _i(p, "fast"), _i(p, "slow"), _i(p, "signal")),
            lambda p: _i(p, "slow") + _i(p, "signal"),
            "MACD line, signal line, histogram",
        ),
        IndicatorType(
            "bbands",
            {"period": 20, "std": 2.0},
            ("middle", "upper", "lower", "width", "percent_b"),
            lambda df, p: bbands(df["close"], _i(p, "period"), _f(p, "std")),
            lambda p: _i(p, "period"),
            "Bollinger Bands (width in % of middle band)",
        ),
        IndicatorType(
            "atr",
            {"period": 14},
            ("value",),
            lambda df, p: _frame(value=atr(df, _i(p, "period"))),
            lambda p: _i(p, "period") + 1,
            "Average true range",
        ),
        IndicatorType(
            "adx",
            {"period": 14},
            ("adx", "plus_di", "minus_di"),
            lambda df, p: adx(df, _i(p, "period")),
            lambda p: _i(p, "period") * 2 + 1,
            "ADX trend strength with +DI / -DI",
        ),
        IndicatorType(
            "stochastic",
            {"k": 14, "d": 3, "smooth": 3},
            ("k", "d"),
            lambda df, p: stochastic(df, _i(p, "k"), _i(p, "d"), _i(p, "smooth")),
            lambda p: _i(p, "k") + _i(p, "d") + _i(p, "smooth"),
            "Slow stochastic %K / %D",
        ),
        IndicatorType(
            "vwap",
            {},
            ("value",),
            lambda df, p: _frame(value=vwap(df)),
            lambda p: 1,
            "Session VWAP (resets daily, UTC)",
        ),
        IndicatorType(
            "supertrend",
            {"period": 10, "multiplier": 3.0},
            ("value", "direction"),
            lambda df, p: supertrend(df, _i(p, "period"), _f(p, "multiplier")),
            lambda p: _i(p, "period") + 1,
            "Supertrend line and direction (+1 / -1)",
        ),
        IndicatorType(
            "swing_points",
            {"left": 3, "right": 3},
            ("high", "low"),
            lambda df, p: _swing_levels(df, _i(p, "left"), _i(p, "right")),
            lambda p: _i(p, "left") + _i(p, "right") + 1,
            "Most recent confirmed swing high / low (forward-filled)",
        ),
        IndicatorType(
            "volume",
            {"period": 20},
            ("value", "sma", "ratio"),
            lambda df, p: volume_profile(df, _i(p, "period")),
            lambda p: _i(p, "period"),
            "Volume, its SMA and the ratio between them",
        ),
    )
}

SUPPORTED_INDICATORS: tuple[str, ...] = tuple(INDICATORS)


def _swing_levels(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    swings = swing_points(df, left, right)
    # A swing is only known ``right`` candles after it happened: shift to avoid look-ahead.
    return pd.DataFrame(
        {
            "high": swings["pivot_high"].shift(right).ffill(),
            "low": swings["pivot_low"].shift(right).ffill(),
        }
    )
