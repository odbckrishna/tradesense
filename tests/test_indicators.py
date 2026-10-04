from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_candles, trending_candles
from tradesense.indicators import technical as ti


def test_ema_of_constant_is_constant() -> None:
    close = pd.Series(1.25, index=range(50), dtype="float64")
    assert ti.ema(close, 10).dropna().eq(1.25).all()


def test_short_inputs_return_nan_not_errors() -> None:
    df = make_candles(8)
    assert ti.atr(df, 14).isna().all()
    assert ti.adx(df, 14).isna().all().all()
    assert ti.rsi(df["close"], 14).isna().all()
    assert ti.macd(df["close"]).isna().all().all()


def test_rsi_bounds_and_trend_bias() -> None:
    up = ti.rsi(trending_candles("up")["close"], 14).dropna()
    down = ti.rsi(trending_candles("down")["close"], 14).dropna()
    assert up.between(0, 100).all()
    assert down.between(0, 100).all()
    assert up.iloc[-50:].mean() > 55 > 45 > down.iloc[-50:].mean()


def test_adx_detects_direction() -> None:
    up = ti.adx(trending_candles("up"), 14).iloc[-1]
    down = ti.adx(trending_candles("down"), 14).iloc[-1]
    assert up["plus_di"] > up["minus_di"]
    assert down["minus_di"] > down["plus_di"]
    assert up["adx"] > 20
    assert not np.isnan(up["adx"])  # the latest value must be computed (ta 0.11 bug)


def test_atr_positive_after_warmup() -> None:
    values = ti.atr(make_candles(100), 14)
    assert values.iloc[:13].isna().all()
    assert (values.dropna() > 0).all()


def test_supertrend_follows_trend() -> None:
    assert ti.supertrend(trending_candles("up"))["direction"].iloc[-1] == 1
    assert ti.supertrend(trending_candles("down"))["direction"].iloc[-1] == -1


def test_crossovers_and_bars_since() -> None:
    a = pd.Series([1.0, 1.0, 3.0, 3.0, 1.0])
    b = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    assert ti.crossed_above(a, b).tolist() == [False, False, True, False, False]
    assert ti.crossed_below(a, b).tolist() == [False, False, False, False, True]
    assert ti.bars_since(ti.crossed_above(a, b)) == 2
    assert ti.bars_since(ti.crossed_above(a, b), lookback=2) is None


def test_session_vwap_resets_daily() -> None:
    df = make_candles(600, "5m", end="2026-03-04 12:00")  # spans more than one UTC day
    line = ti.vwap(df)
    first_of_day = df.index[df.index.floor("D") == df.index[-1].floor("D")][0]
    row = df.loc[first_of_day]
    expected = (row["high"] + row["low"] + row["close"]) / 3
    assert line.loc[first_of_day] == pytest.approx(expected)


def test_swing_points_have_no_lookahead_in_rule_levels() -> None:
    df = make_candles(120)
    levels = ti.INDICATORS["swing_points"].run(df, {"left": 3, "right": 3})
    raw = ti.swing_points(df, 3, 3)
    # a swing confirmed at bar i must not be visible before bar i + right
    first_swing = raw["pivot_high"].first_valid_index()
    assert first_swing is not None
    position = df.index.get_loc(first_swing)
    assert levels["high"].iloc[: position + 3].isna().all()


def test_pivot_levels_need_complete_previous_period() -> None:
    df = make_candles(299, "1m")
    hourly = ti.pivot_levels(df, "1h")
    assert hourly is not None
    assert hourly["s1"] < hourly["pivot"] < hourly["r1"]
    assert ti.pivot_levels(df, "1D") is None  # 5 hours of data has no complete previous day
    with pytest.raises(ValueError, match="Invalid pivot period"):
        ti.pivot_levels(df, "banana")


def test_infer_bar_seconds_is_unit_safe() -> None:
    assert ti.infer_bar_seconds(make_candles(10, "15m").index) == 900


def test_indicator_registry_validates_params() -> None:
    ema = ti.INDICATORS["ema"]
    assert ema.resolve_params({"period": 10.0}) == {"period": 10}
    with pytest.raises(ValueError, match="no parameter 'length'"):
        ema.resolve_params({"length": 5})
    with pytest.raises(ValueError, match="must be whole"):
        ema.resolve_params({"period": 10.5})
    with pytest.raises(ValueError, match="out of range"):
        ema.resolve_params({"period": 0})
    frame = ti.INDICATORS["macd"].run(make_candles(100), {})
    assert list(frame.columns) == ["macd", "signal", "hist"]
    assert set(ti.SUPPORTED_INDICATORS) == {
        "ema",
        "sma",
        "rsi",
        "macd",
        "bbands",
        "atr",
        "adx",
        "stochastic",
        "vwap",
        "supertrend",
        "swing_points",
        "volume",
    }


def test_atr_matches_ta_average_true_range() -> None:
    """The vectorised ATR must reproduce ta's loop-based implementation exactly."""
    import ta

    for seed in (1, 2):
        df = make_candles(400, "1m", seed=seed)
        for period in (5, 14):
            for view in (df, df.iloc[-120:], df.iloc[: period + 1]):
                reference = ta.volatility.AverageTrueRange(
                    view["high"], view["low"], view["close"], window=period
                ).average_true_range()
                pd.testing.assert_series_equal(
                    ti.atr(view, period),
                    reference.where(reference > 0).astype("float64"),
                    check_names=False,
                    rtol=1e-12,
                )
