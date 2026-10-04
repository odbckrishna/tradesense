"""YAML rule strategies.

A rule file declares indicators and two boolean expressions (``buy`` / ``sell``);
anything else is SIDEWAYS. Example (``config/rules/rsi_ema_combo.yml``)::

    name: rsi_ema_combo
    description: RSI pullback in direction of EMA trend
    category: momentum
    min_candles: 60
    indicators:
      - { id: ema_fast, type: ema, period: 20 }
      - { id: ema_slow, type: ema, period: 50 }
      - { id: rsi,      type: rsi, period: 14 }
    rules:
      buy:  "ema_fast > ema_slow and rsi < 40"
      sell: "ema_fast < ema_slow and rsi > 60"

Expressions are parsed with :mod:`ast`, checked against a small whitelist and then
interpreted by :class:`SafeEvaluator` over pandas Series. ``eval``/``exec`` are never
used. Allowed: numbers, ``True``/``False``, indicator ids, ``<id>.<output>``,
price columns (``open high low close volume``), arithmetic ``+ - * /``, comparisons,
``and or not`` and the functions in :data:`FUNCTIONS`.
"""

from __future__ import annotations

import ast
import asyncio
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from tradesense.indicators.technical import (
    INDICATORS,
    ago,
    bars_since,
    crossed_above,
    crossed_below,
)
from tradesense.strategies.base import (
    FeatureBuilder,
    Strategy,
    StrategyCategory,
    StrategyContext,
    StrategyFeatures,
    StrategySource,
)

MAX_EXPRESSION_LENGTH = 500
MAX_EXPRESSION_NODES = 120
MAX_WINDOW = 500
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,39}$")

type Value = pd.Series | float | bool


class RuleValidationError(ValueError):
    """A rule file or expression is malformed or unsafe."""


_ACRONYMS = {"rsi", "ema", "sma", "macd", "atr", "adx", "vwap", "bb", "di", "tk", "sr"}


def humanize(name: str) -> str:
    """``rsi_ema_combo`` -> ``RSI EMA Combo``."""
    return " ".join(w.upper() if w in _ACRONYMS else w.capitalize() for w in name.split("_") if w)


# --------------------------------------------------------------------------- functions


@dataclass(frozen=True)
class RuleFunction:
    min_args: int
    max_args: int
    #: Positions of arguments that must be integer literals (window sizes).
    int_args: tuple[int, ...]
    impl: Callable[..., Value]


def _series(value: Value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(float(value), index=index)


def _shift(x: Value, n: int = 1) -> Value:
    return x.shift(n) if isinstance(x, pd.Series) else x


def _rising(x: Value, n: int = 1) -> Value:
    return (x > x.shift(n)) if isinstance(x, pd.Series) else False


def _falling(x: Value, n: int = 1) -> Value:
    return (x < x.shift(n)) if isinstance(x, pd.Series) else False


def _highest(x: Value, n: int) -> Value:
    return x.rolling(n, min_periods=n).max() if isinstance(x, pd.Series) else x


def _lowest(x: Value, n: int) -> Value:
    return x.rolling(n, min_periods=n).min() if isinstance(x, pd.Series) else x


def _abs(x: Value) -> Value:
    return x.abs() if isinstance(x, pd.Series) else abs(float(x))


def _pair(a: Value, b: Value) -> pd.DataFrame | None:
    """Both operands as aligned columns, or None when both are scalars."""
    for value in (a, b):
        if isinstance(value, pd.Series):
            index = cast(pd.Series, value).index
            return pd.concat([_series(a, index), _series(b, index)], axis=1)
    return None


def _min(a: Value, b: Value) -> Value:
    pair = _pair(a, b)
    return pair.min(axis=1) if pair is not None else min(float(a), float(b))


def _max(a: Value, b: Value) -> Value:
    pair = _pair(a, b)
    return pair.max(axis=1) if pair is not None else max(float(a), float(b))


def _cross_up(a: Value, b: Value) -> Value:
    if not isinstance(a, pd.Series):
        raise RuleValidationError("crossed_above() needs an indicator as first argument")
    return crossed_above(a, b if isinstance(b, pd.Series) else float(b))


def _cross_down(a: Value, b: Value) -> Value:
    if not isinstance(a, pd.Series):
        raise RuleValidationError("crossed_below() needs an indicator as first argument")
    return crossed_below(a, b if isinstance(b, pd.Series) else float(b))


FUNCTIONS: dict[str, RuleFunction] = {
    "crossed_above": RuleFunction(2, 2, (), _cross_up),
    "crossed_below": RuleFunction(2, 2, (), _cross_down),
    "prev": RuleFunction(1, 2, (1,), _shift),
    "rising": RuleFunction(1, 2, (1,), _rising),
    "falling": RuleFunction(1, 2, (1,), _falling),
    "highest": RuleFunction(2, 2, (1,), _highest),
    "lowest": RuleFunction(2, 2, (1,), _lowest),
    "abs": RuleFunction(1, 1, (), _abs),
    "min": RuleFunction(2, 2, (), _min),
    "max": RuleFunction(2, 2, (), _max),
}

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_CMP_OPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    *_BIN_OPS,
    ast.Compare,
    *_CMP_OPS,
    ast.Name,
    ast.Attribute,
    ast.Constant,
    ast.Call,
    ast.Load,
)


# --------------------------------------------------------------------------- validation


def parse_expression(expression: str, outputs: Mapping[str, tuple[str, ...]]) -> ast.Expression:
    """Parse and whitelist-check a rule expression.

    ``outputs`` maps each declared indicator id to its output names.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise RuleValidationError("expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise RuleValidationError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise RuleValidationError(f"syntax error in '{expression}': {exc.msg}") from None

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_EXPRESSION_NODES:
        raise RuleValidationError("expression is too complex")
    call_names = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODES):
            raise RuleValidationError(f"'{type(node).__name__}' is not allowed in rule expressions")
        if isinstance(node, ast.Constant):
            # bool is a subclass of int, so True/False pass; strings, None, bytes do not.
            if not isinstance(node.value, int | float):
                raise RuleValidationError("only numbers and True/False constants are allowed")
        elif isinstance(node, ast.Name):
            if id(node) in call_names:
                if node.id not in FUNCTIONS:
                    raise RuleValidationError(f"function '{node.id}' is not allowed")
            elif node.id not in outputs and node.id not in PRICE_COLUMNS:
                raise RuleValidationError(f"unknown name '{node.id}'")
        elif isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id not in outputs:
                raise RuleValidationError(
                    "attribute access is only allowed as <indicator>.<output>"
                )
            if node.attr not in outputs[node.value.id]:
                raise RuleValidationError(
                    f"indicator '{node.value.id}' has no output '{node.attr}' "
                    f"(available: {', '.join(outputs[node.value.id])})"
                )
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise RuleValidationError("only whitelisted functions may be called")
            if node.keywords:
                raise RuleValidationError("keyword arguments are not supported")
            spec = FUNCTIONS[node.func.id]
            if not spec.min_args <= len(node.args) <= spec.max_args:
                raise RuleValidationError(
                    f"{node.func.id}() takes {spec.min_args}-{spec.max_args} arguments"
                )
            for position in spec.int_args:
                if position < len(node.args):
                    arg = node.args[position]
                    if not (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, int)
                        and not isinstance(arg.value, bool)
                        and 1 <= arg.value <= MAX_WINDOW
                    ):
                        raise RuleValidationError(
                            f"{node.func.id}() window must be an integer between 1 and {MAX_WINDOW}"
                        )
    return tree


class SafeEvaluator:
    """Interprets a validated expression tree over a namespace of Series."""

    def __init__(self, namespace: Mapping[str, pd.Series], index: pd.Index) -> None:
        self.namespace = namespace
        self.index = index

    def evaluate(self, tree: ast.Expression) -> pd.Series:
        result = self._eval(tree.body)
        series = _series(result, self.index) if not isinstance(result, pd.Series) else result
        if series.dtype != bool:
            series = series.astype("boolean").fillna(False).astype(bool)
        return series

    def _eval(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.namespace[node.id]
        if isinstance(node, ast.Attribute):
            assert isinstance(node.value, ast.Name)
            return self.namespace[f"{node.value.id}.{node.attr}"]
        if isinstance(node, ast.BoolOp):
            values = [self._as_bool(self._eval(v)) for v in node.values]
            result = values[0]
            for value in values[1:]:
                result = (result & value) if isinstance(node.op, ast.And) else (result | value)
            return result
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return ~self._as_bool(operand)
            if isinstance(node.op, ast.USub):
                return -operand
            return operand
        if isinstance(node, ast.BinOp):
            left, right = self._eval(node.left), self._eval(node.right)
            with np.errstate(divide="ignore", invalid="ignore"):
                return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            combined: pd.Series | None = None
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._eval(comparator)
                part = self._as_bool(_CMP_OPS[type(op)](left, right))
                combined = part if combined is None else (combined & part)
                left = right
            assert combined is not None
            return combined
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            spec = FUNCTIONS[node.func.id]
            args = [
                node.args[i].value  # type: ignore[attr-defined]
                if i in spec.int_args
                else self._eval(node.args[i])
                for i in range(len(node.args))
            ]
            return spec.impl(*args)
        raise RuleValidationError(f"unsupported node {type(node).__name__}")  # pragma: no cover

    def _as_bool(self, value: Value) -> pd.Series:
        if isinstance(value, pd.Series):
            if value.dtype == bool:
                return value
            return value.astype("boolean").fillna(False).astype(bool)
        return pd.Series(bool(value), index=self.index)


# --------------------------------------------------------------------------- rule spec


class IndicatorDecl(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError(f"indicator id '{value}' must be lower_snake_case")
        if value in PRICE_COLUMNS or value in FUNCTIONS or value in {"and", "or", "not"}:
            raise ValueError(f"indicator id '{value}' is reserved")
        return value

    @field_validator("type")
    @classmethod
    def _type(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in INDICATORS:
            raise ValueError(
                f"unknown indicator type '{value}'; supported: {', '.join(INDICATORS)}"
            )
        return value

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class RuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    buy: str
    sell: str


class RuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    category: StrategyCategory
    min_candles: int = Field(default=30, ge=1, le=5000)
    display_name: str | None = None
    indicators: list[IndicatorDecl] = Field(min_length=1)
    rules: RuleSet

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError("name must be lower_snake_case")
        return value

    @model_validator(mode="after")
    def _check(self) -> RuleSpec:
        ids = [ind.id for ind in self.indicators]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate indicator ids: {', '.join(sorted(duplicates))}")
        for ind in self.indicators:
            INDICATORS[ind.type].resolve_params(ind.params)  # raises ValueError if invalid
        return self


# --------------------------------------------------------------------------- strategy


class RuleStrategy(Strategy):
    """A strategy defined by a YAML rule file."""

    source: StrategySource = "rule"

    def __init__(self, spec: RuleSpec, path: Path | None = None) -> None:
        self.spec = spec
        self.path = path
        self.name = spec.name
        self.display_name = spec.display_name or humanize(spec.name)
        self.description = spec.description
        self.category = spec.category
        self.min_candles = spec.min_candles
        self.outputs = {ind.id: INDICATORS[ind.type].outputs for ind in spec.indicators}
        self.default_params = {
            f"{ind.id}.{key}": value
            for ind in spec.indicators
            for key, value in INDICATORS[ind.type].resolve_params(ind.params).items()
        }
        self.buy_tree = parse_expression(spec.rules.buy, self.outputs)
        self.sell_tree = parse_expression(spec.rules.sell, self.outputs)

    def indicator_params(self, params: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        per_indicator: dict[str, dict[str, Any]] = {ind.id: {} for ind in self.spec.indicators}
        for key, value in params.items():
            ind_id, _, param = key.partition(".")
            if ind_id in per_indicator and param:
                per_indicator[ind_id][param] = value
        return per_indicator

    def required_candles(self, params: Mapping[str, Any]) -> int:
        per_indicator = self.indicator_params(self.merge_params(params))
        warmups = [
            INDICATORS[ind.type].warmup(INDICATORS[ind.type].resolve_params(per_indicator[ind.id]))
            for ind in self.spec.indicators
        ]
        return max([self.min_candles, *(w + 1 for w in warmups)])

    async def analyze(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        timeframe: str,
        context: StrategyContext,
    ) -> StrategyFeatures:
        builder = FeatureBuilder(self, df, params, timeframe, context.history_length)
        await asyncio.to_thread(self._evaluate, df, params, builder)
        return builder.build()

    def _namespace(
        self, df: pd.DataFrame, params: Mapping[str, Any], fb: FeatureBuilder | None = None
    ) -> dict[str, pd.Series]:
        """Every name a rule expression can use, optionally recorded into ``fb``."""
        per_indicator = self.indicator_params(params)
        namespace: dict[str, pd.Series] = {col: df[col] for col in PRICE_COLUMNS}
        for ind in self.spec.indicators:
            kind = INDICATORS[ind.type]
            frame = kind.run(df, per_indicator[ind.id])
            for output in kind.outputs:
                namespace[f"{ind.id}.{output}"] = frame[output]
            namespace[ind.id] = frame[kind.primary]
            if fb is not None:
                fb.series(ind.id, frame[kind.primary])
                for output in kind.outputs[1:]:
                    fb.series(f"{ind.id}.{output}", frame[output], history=False)
        return namespace

    def masks(
        self, df: pd.DataFrame, params: Mapping[str, Any], fb: FeatureBuilder | None = None
    ) -> tuple[pd.Series, pd.Series]:
        """BUY and SELL rule truth for every candle (vectorised: live and backtests)."""
        evaluator = SafeEvaluator(self._namespace(df, params, fb), df.index)
        return evaluator.evaluate(self.buy_tree), evaluator.evaluate(self.sell_tree)

    def signal_series(self, df: pd.DataFrame, params: Mapping[str, Any] | None = None) -> pd.Series:
        """+1 BUY, -1 SELL, 0 SIDEWAYS (neither or both rules true) per candle; NaN until
        enough candles exist for the live strategy to run."""
        merged = self.merge_params(params)
        buy, sell = self.masks(df, merged)
        b, s = buy.to_numpy(dtype=bool), sell.to_numpy(dtype=bool)
        values = np.where(b & ~s, 1.0, np.where(s & ~b, -1.0, 0.0))
        values[: max(0, self.required_candles(merged) - 1)] = np.nan
        return pd.Series(values, index=df.index)

    def _evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        buy, sell = self.masks(df, params, fb)
        fb.series("close", df["close"])
        buy_now, sell_now = bool(buy.iloc[-1]), bool(sell.iloc[-1])
        fb.context["rules"] = {"buy": self.spec.rules.buy, "sell": self.spec.rules.sell}
        fb.value("buy_rule_active", buy_now)
        fb.value("sell_rule_active", sell_now)

        for label, mask, now in (("BUY", buy, buy_now), ("SELL", sell, sell_now)):
            if now:
                streak = _streak(mask)
                started = streak - 1
                fb.event(
                    f"{label} rule true on the latest candle "
                    f"(true for {streak} candle{'s' if streak != 1 else ''}, "
                    f"became true {ago(started)})"
                )
            else:
                last = bars_since(mask, 50)
                if last is not None:
                    fb.event(f"{label} rule was last true {ago(last)}")

        if buy_now and sell_now:
            fb.signal("SIDEWAYS", "Both BUY and SELL rules are true (conflict)")
        elif buy_now:
            fb.signal("BUY", f"BUY rule satisfied: {self.spec.rules.buy}")
        elif sell_now:
            fb.signal("SELL", f"SELL rule satisfied: {self.spec.rules.sell}")
        else:
            fb.signal("SIDEWAYS", "Neither BUY nor SELL rule is satisfied")


def _streak(mask: pd.Series) -> int:
    values = mask.to_numpy(dtype=bool)[::-1]
    if not values.size or not values[0]:
        return 0
    falses = np.flatnonzero(~values)
    return int(falses[0]) if falses.size else int(values.size)


# --------------------------------------------------------------------------- loading


def load_rule(data: Any, source: str = "<rule>") -> RuleStrategy:
    if not isinstance(data, dict):
        raise RuleValidationError(f"{source}: a rule file must contain a mapping")
    try:
        spec = RuleSpec.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise RuleValidationError(f"{source}: {details}") from None
    try:
        return RuleStrategy(spec)
    except RuleValidationError as exc:
        raise RuleValidationError(f"{source}: {exc}") from None


def load_rule_file(path: Path) -> RuleStrategy:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuleValidationError(f"{path.name}: invalid YAML: {exc}") from None
    except OSError as exc:
        raise RuleValidationError(f"{path.name}: cannot read file: {exc}") from None
    strategy = load_rule(data, path.name)
    strategy.path = path
    return strategy


def load_rules_dir(directory: Path) -> tuple[list[RuleStrategy], list[str]]:
    """Load every ``*.yml`` / ``*.yaml`` rule. Invalid files are reported, never fatal."""
    strategies: list[RuleStrategy] = []
    errors: list[str] = []
    if not directory.is_dir():
        return strategies, errors
    seen: set[str] = set()
    for path in sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")]):
        try:
            strategy = load_rule_file(path)
        except RuleValidationError as exc:
            errors.append(str(exc))
            continue
        if strategy.name in seen:
            errors.append(f"{path.name}: duplicate rule name '{strategy.name}'")
            continue
        seen.add(strategy.name)
        strategies.append(strategy)
    return strategies, errors
