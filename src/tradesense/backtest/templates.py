"""Rule templates the backtest searches to build a new strategy for an instrument.

Each template is a classic setup with a small parameter grid, expressed as an ordinary
YAML rule (:mod:`tradesense.strategies.rule_engine`), so a winner can be saved to
``config/rules/`` and run live through the same engine it was backtested with.
Candidate names are short keys (``emax_9_21``); saving prefixes instrument and timeframe.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

from tradesense.strategies.rule_engine import RuleStrategy, load_rule

Spec = dict[str, Any]


def _num(value: float) -> str:
    """``2.5`` -> ``2p5`` (rule names are lower_snake_case)."""
    return f"{value:g}".replace(".", "p")


def _spec(
    name: str,
    display: str,
    category: str,
    description: str,
    indicators: list[Spec],
    buy: str,
    sell: str,
) -> Spec:
    return {
        "name": name,
        "display_name": display,
        "description": description,
        "category": category,
        "indicators": indicators,
        "rules": {"buy": buy, "sell": sell},
    }


def _ema_cross() -> Iterator[Spec]:
    for fast, slow in product((5, 9, 12, 20), (21, 34, 50, 100)):
        yield _spec(
            f"emax_{fast}_{slow}",
            f"EMA {fast}/{slow} trend",
            "trend",
            f"Long while EMA{fast} is above EMA{slow} and price holds above EMA{fast}; "
            "short on the mirror image.",
            [
                {"id": "ema_fast", "type": "ema", "period": fast},
                {"id": "ema_slow", "type": "ema", "period": slow},
            ],
            "ema_fast > ema_slow and close > ema_fast",
            "ema_fast < ema_slow and close < ema_fast",
        )


def _ema_rsi_pullback() -> Iterator[Spec]:
    for trend, period, level in product((50, 100, 200), (7, 14), (35, 40, 45)):
        yield _spec(
            f"emarsi_{trend}_{period}_{level}",
            f"EMA{trend} trend + RSI{period} pullback",
            "momentum",
            f"Buy RSI{period} dips below {level} while price is above EMA{trend}; sell "
            f"rallies above {100 - level} below it.",
            [
                {"id": "trend", "type": "ema", "period": trend},
                {"id": "rsi", "type": "rsi", "period": period},
            ],
            f"close > trend and rsi < {level}",
            f"close < trend and rsi > {100 - level}",
        )


def _rsi_reversion() -> Iterator[Spec]:
    for period, level in product((7, 14, 21), (20, 25, 30)):
        yield _spec(
            f"rsix_{period}_{level}",
            f"RSI{period} cross-back {level}/{100 - level}",
            "mean_reversion",
            f"Buy when RSI{period} crosses back above {level}; sell when it crosses back "
            f"below {100 - level}.",
            [{"id": "rsi", "type": "rsi", "period": period}],
            f"crossed_above(rsi, {level})",
            f"crossed_below(rsi, {100 - level})",
        )
        yield _spec(
            f"rsiturn_{period}_{level}",
            f"RSI{period} turn {level}/{100 - level}",
            "mean_reversion",
            f"Buy while RSI{period} is below {level} and rising; sell while above "
            f"{100 - level} and falling.",
            [{"id": "rsi", "type": "rsi", "period": period}],
            f"rsi < {level} and rising(rsi)",
            f"rsi > {100 - level} and falling(rsi)",
        )


def _macd_adx() -> Iterator[Spec]:
    for (fast, slow, signal), level in product(((12, 26, 9), (5, 35, 5)), (20, 25, 30)):
        yield _spec(
            f"macdadx_{fast}_{slow}_{level}",
            f"MACD {fast}/{slow} + ADX>{level}",
            "trend",
            f"MACD histogram direction confirmed by ADX above {level} and the leading DI.",
            [
                {"id": "macd", "type": "macd", "fast": fast, "slow": slow, "signal": signal},
                {"id": "adx", "type": "adx", "period": 14},
            ],
            f"macd.hist > 0 and adx > {level} and adx.plus_di > adx.minus_di",
            f"macd.hist < 0 and adx > {level} and adx.minus_di > adx.plus_di",
        )


def _bollinger_reversion() -> Iterator[Spec]:
    for period, std, level in product((14, 20), (2.0, 2.5), (30, 35)):
        yield _spec(
            f"bbrev_{period}_{_num(std)}_{level}",
            f"Bollinger {period}/{std:g} reversion + RSI",
            "mean_reversion",
            f"Buy closes below the lower band with RSI under {level}; sell closes above the "
            f"upper band with RSI over {100 - level}.",
            [
                {"id": "bb", "type": "bbands", "period": period, "std": std},
                {"id": "rsi", "type": "rsi", "period": 14},
            ],
            f"close < bb.lower and rsi < {level}",
            f"close > bb.upper and rsi > {100 - level}",
        )


def _supertrend_rsi() -> Iterator[Spec]:
    for period, multiplier, gate in product((7, 10, 14), (2.0, 3.0), (50, 55)):
        yield _spec(
            f"strsi_{period}_{_num(multiplier)}_{gate}",
            f"Supertrend {period}/{multiplier:g} + RSI {gate}",
            "trend",
            f"Supertrend direction confirmed by RSI above {gate} (below {100 - gate}).",
            [
                {"id": "st", "type": "supertrend", "period": period, "multiplier": multiplier},
                {"id": "rsi", "type": "rsi", "period": 14},
            ],
            f"st.direction > 0 and rsi > {gate}",
            f"st.direction < 0 and rsi < {100 - gate}",
        )


def _stochastic_trend() -> Iterator[Spec]:
    for k, trend, zone in product((5, 14), (50, 100), (20, 30)):
        yield _spec(
            f"stoch_{k}_{trend}_{zone}",
            f"Stochastic {k} cross with EMA{trend} trend",
            "momentum",
            f"%K crossing %D from below {zone} with price above EMA{trend}, and the mirror.",
            [
                {"id": "stoch", "type": "stochastic", "k": k, "d": 3, "smooth": 3},
                {"id": "trend", "type": "ema", "period": trend},
            ],
            f"close > trend and crossed_above(stoch.k, stoch.d) and stoch.d < {zone}",
            f"close < trend and crossed_below(stoch.k, stoch.d) and stoch.d > {100 - zone}",
        )


def _breakout() -> Iterator[Spec]:
    for lookback, margin in product((10, 20, 34, 55), (0.0, 0.25)):
        yield _spec(
            f"brk_{lookback}_{_num(margin)}",
            f"{lookback}-candle breakout" + (f" +{margin:g} ATR" if margin else ""),
            "volatility",
            f"Close beyond the {lookback}-candle high / low"
            + (f" by {margin:g} ATR." if margin else "."),
            [{"id": "atr", "type": "atr", "period": 14}],
            f"close > prev(highest(high, {lookback}))" + (f" + {margin:g} * atr" if margin else ""),
            f"close < prev(lowest(low, {lookback}))" + (f" - {margin:g} * atr" if margin else ""),
        )


def _vwap_trend() -> Iterator[Spec]:
    for period in (9, 20, 34):
        yield _spec(
            f"vwap_{period}",
            f"VWAP trend + EMA{period}",
            "levels",
            f"Price and a rising EMA{period} above the session VWAP; the mirror for sells.",
            [{"id": "vwap", "type": "vwap"}, {"id": "ema", "type": "ema", "period": period}],
            "close > vwap and ema > vwap and rising(ema)",
            "close < vwap and ema < vwap and falling(ema)",
        )


TEMPLATES = (
    _ema_cross,
    _ema_rsi_pullback,
    _rsi_reversion,
    _macd_adx,
    _bollinger_reversion,
    _supertrend_rsi,
    _stochastic_trend,
    _breakout,
    _vwap_trend,
)


def template_specs() -> list[Spec]:
    return [spec for template in TEMPLATES for spec in template()]


def template_candidates() -> list[RuleStrategy]:
    """Every template expanded over its grid, validated by the rule engine."""
    return [load_rule(spec, f"template {spec['name']}") for spec in template_specs()]
