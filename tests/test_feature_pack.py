from __future__ import annotations

import io

from PIL import Image

from tests.helpers import make_candles
from tradesense.strategies.chart_renderer import render_chart, render_chart_async
from tradesense.strategies.feature_pack import compute_feature_pack, compute_required_indicators


def test_feature_pack_contents() -> None:
    pack = compute_feature_pack(make_candles(299, "5m"))
    latest = pack["latest"]
    for key in (
        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "atr14",
        "adx14",
        "plus_di",
        "minus_di",
    ):
        assert latest[key] is not None, key
    assert set(pack["history"]) == {"close", "ema20", "rsi14", "macd_hist"}
    assert all(len(v) == 10 for v in pack["history"].values())
    assert pack["swings"]["highs"]
    assert pack["swings"]["lows"]
    assert pack["swings"]["highs"][-1]["candles_ago"] >= 3  # swings need confirmation
    assert pack["volume"]["ratio_to_sma20"] is not None
    assert any(line.startswith("RSI14") for line in pack["summary"])


def test_feature_pack_without_enough_history_for_ema200() -> None:
    pack = compute_feature_pack(make_candles(120, "1m"))
    assert pack["latest"]["ema200"] is None
    assert "EMA200 not available (fewer than 200 candles)" in pack["summary"]


def test_required_indicators() -> None:
    values = compute_required_indicators(make_candles(200), ["macd", "vwap", "unknown"])
    assert set(values) == {"macd", "vwap"}
    assert set(values["macd"]["latest"]) == {"macd", "signal", "hist"}
    assert len(values["vwap"]["history"]) == 10


def test_chart_renderer_png() -> None:
    png = render_chart(make_candles(300), title="EUR_USD 5m", indicators=["rsi", "bbands"])
    assert png.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(png)) as image:
        assert image.size == (1200, 720)


async def test_chart_renderer_async_handles_short_history() -> None:
    png = await render_chart_async(make_candles(40), indicators=["macd", "swing_points"])
    assert png.startswith(b"\x89PNG")
