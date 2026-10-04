from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.helpers import REPO_ROOT, make_candles, trending_candles
from tradesense.strategies.rule_engine import (
    RuleValidationError,
    SafeEvaluator,
    load_rule,
    load_rule_file,
    load_rules_dir,
    parse_expression,
)

OUTPUTS = {"rsi": ("value",), "fast": ("value",), "macd": ("macd", "signal", "hist")}


def _rule(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "test_rule",
        "description": "test",
        "category": "momentum",
        "min_candles": 30,
        "indicators": [
            {"id": "fast", "type": "ema", "period": 5},
            {"id": "slow", "type": "ema", "period": 20},
            {"id": "rsi", "type": "rsi", "period": 14},
        ],
        "rules": {"buy": "fast > slow and rsi > 50", "sell": "fast < slow and rsi < 50"},
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- loading


def test_example_rule_file_loads() -> None:
    strategy = load_rule_file(REPO_ROOT / "config" / "rules" / "rsi_ema_combo.yml")
    assert strategy.name == "rsi_ema_combo"
    assert strategy.source == "rule"
    assert strategy.category == "momentum"
    assert strategy.default_params == {
        "ema_fast.period": 20,
        "ema_slow.period": 50,
        "rsi.period": 14,
    }
    assert strategy.required_candles(strategy.default_params) == 60


async def test_rule_evaluates_trend() -> None:
    strategy = load_rule(_rule())
    up = await strategy.compute(trending_candles("up"), {}, "5m")
    down = await strategy.compute(trending_candles("down"), {}, "5m")
    assert up.status == "ok"
    assert up.pre_signal == "BUY"
    assert down.pre_signal == "SELL"
    assert up.latest["buy_rule_active"] is True
    assert any("BUY rule true on the latest candle" in e for e in up.events)
    assert up.context["rules"]["buy"] == "fast > slow and rsi > 50"
    assert len(up.history["rsi"]) == 10


async def test_rule_params_can_be_overridden_per_timeframe() -> None:
    strategy = load_rule(_rule())
    features = await strategy.compute(make_candles(200), {"slow.period": 100}, "15m")
    assert features.params["slow.period"] == 100
    assert strategy.required_candles({"slow.period": 150}) == 151


def test_malformed_yaml_rule_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.yml"
    path.write_text("name: broken\nindicators: [unclosed\n", encoding="utf-8")
    with pytest.raises(RuleValidationError, match="invalid YAML"):
        load_rule_file(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rules": {"buy": "fast >"}}, "rules.sell"),  # missing sell + bad buy
        ({"rules": {"buy": "fast > ", "sell": "rsi < 50"}}, "syntax error"),
        ({"indicators": [{"id": "x", "type": "fourier"}]}, "unknown indicator type"),
        ({"indicators": [{"id": "x", "type": "ema", "length": 5}]}, "no parameter 'length'"),
        ({"category": "astrology"}, "category"),
        ({"rules": {"buy": "ghost > 1", "sell": "rsi < 50"}}, "unknown name 'ghost'"),
        ({"rules": {"buy": "rsi.hist > 1", "sell": "rsi < 50"}}, "has no output 'hist'"),
        ({"indicators": [{"id": "close", "type": "ema"}]}, "reserved"),
    ],
)
def test_malformed_rules_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(RuleValidationError, match=message):
        load_rule(_rule(**overrides))


def test_rules_dir_skips_invalid_files(tmp_path: Path) -> None:
    (tmp_path / "good.yml").write_text(
        (REPO_ROOT / "config" / "rules" / "rsi_ema_combo.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "bad.yml").write_text("name: bad\n", encoding="utf-8")
    (tmp_path / "evil.yml").write_text(
        "name: evil\ndescription: x\ncategory: trend\n"
        "indicators: [{id: e, type: ema}]\n"
        "rules: {buy: \"__import__('os').system('echo pwned')\", sell: 'e > 1'}\n",
        encoding="utf-8",
    )
    strategies, errors = load_rules_dir(tmp_path)
    assert [s.name for s in strategies] == ["rsi_ema_combo"]
    assert len(errors) == 2
    assert any("bad.yml" in e for e in errors)
    assert any("evil.yml" in e for e in errors)


# --------------------------------------------------------------------------- safety


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",
        "open('secrets.txt')",
        "rsi.__class__.__bases__",
        "(lambda: 1)()",
        "[x for x in (1, 2)]",
        "rsi if rsi > 1 else 0",
        "close[0] > 1",
        "rsi ** 99999999",
        "'text' == 'text'",
        "eval('1')",
        "rsi > 50; import os",
        "(y := 5) > 1",
        "f'{rsi}' == 1",
        "prev(rsi, 10000) > 1",
        "prev(rsi, fast) > 1",
        "abs(x=rsi) > 1",
        "rsi > " + "+".join(["1"] * 200),
    ],
)
def test_unsafe_expressions_rejected(expression: str) -> None:
    with pytest.raises(RuleValidationError):
        parse_expression(expression, OUTPUTS)


def test_expression_length_limit() -> None:
    with pytest.raises(RuleValidationError, match="longer than"):
        parse_expression("rsi > 1 and " * 60 + "rsi > 1", OUTPUTS)


def test_safe_evaluator_semantics() -> None:
    index = pd.RangeIndex(5)
    namespace = {
        "rsi": pd.Series([20.0, 25.0, 35.0, 60.0, 75.0], index=index),
        "fast": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index),
        "macd.macd": pd.Series([-1.0, -0.5, 0.2, 0.4, 0.3], index=index),
        "macd.signal": pd.Series([0.0, 0.0, 0.0, 0.1, 0.35], index=index),
        "macd.hist": pd.Series([-1.0, -0.5, 0.2, 0.3, -0.05], index=index),
        "close": pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=index),
    }
    namespace["macd"] = namespace["macd.macd"]

    def run(expr: str) -> list[bool]:
        tree = parse_expression(expr, {**OUTPUTS})
        return SafeEvaluator(namespace, index).evaluate(tree).tolist()

    assert run("rsi > 30 and rsi < 70") == [False, False, True, True, False]
    assert run("30 < rsi < 70") == [False, False, True, True, False]
    assert run("not rsi > 30") == [True, True, False, False, False]
    assert run("crossed_above(macd, macd.signal)") == [False, False, True, False, False]
    assert run("crossed_below(macd.hist, 0)") == [False, False, False, False, True]
    assert run("rising(rsi) and prev(rsi, 1) > 30") == [False, False, False, True, True]
    assert run("fast * 2 - 1 >= highest(fast, 3)") == [False, False, True, True, True]
    assert run("abs(macd.hist) > 0.4 or rsi >= 75") == [True, True, False, False, True]
    assert run("True") == [True] * 5
