"""Strategy base classes.

Strategies are timeframe-agnostic: they analyse whatever candle DataFrame they
receive. The timeframe is passed only as context (for the LLM prompt and for
per-timeframe parameter overrides). Strategies compute FEATURES; the LLM makes
the DECISION.
"""

from __future__ import annotations

import asyncio
import base64
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tradesense.config import StrategyParamsConfig

StrategySource = Literal["builtin", "rule", "document", "image"]
StrategyCategory = Literal["trend", "momentum", "mean_reversion", "volatility", "levels", "pattern"]
Signal = Literal["BUY", "SELL", "SIDEWAYS"]
FeatureStatus = Literal["ok", "insufficient_data", "error"]

CATEGORIES: tuple[StrategyCategory, ...] = (
    "trend",
    "momentum",
    "mean_reversion",
    "volatility",
    "levels",
    "pattern",
)


class StrategyError(RuntimeError):
    """A strategy could not be evaluated."""


class InsufficientDataError(StrategyError):
    """Raised during evaluation when the data cannot support the analysis."""

    def __init__(self, message: str, needed: int | None = None) -> None:
        super().__init__(message)
        self.needed = needed


class UnknownStrategyError(KeyError):
    """No strategy with that name is registered (or it is not approved)."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return f"Unknown strategy '{self.name}'"


# --------------------------------------------------------------------------- models


class ImageAttachment(BaseModel):
    """An image sent to a vision model (reference chart or rendered live chart)."""

    label: str
    media_type: str
    data_b64: str

    @classmethod
    def from_bytes(cls, label: str, data: bytes, media_type: str) -> ImageAttachment:
        return cls(label=label, media_type=media_type, data_b64=base64.b64encode(data).decode())

    def to_bytes(self) -> bytes:
        return base64.b64decode(self.data_b64)


class StrategyFeatures(BaseModel):
    """Everything a strategy computed for one timeframe: the input to the LLM."""

    strategy: str
    display_name: str
    description: str
    source: StrategySource
    category: StrategyCategory
    timeframe: str
    status: FeatureStatus = "ok"
    message: str | None = None
    candles_available: int
    candles_required: int
    params: dict[str, Any] = Field(default_factory=dict)
    latest: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    history: dict[str, list[float | None]] = Field(default_factory=dict)
    events: list[str] = Field(default_factory=list)
    pre_signal: Signal | None = None
    pre_signal_reason: str | None = None
    images: list[ImageAttachment] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    last_candle_time: datetime | None = None
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StrategyInfo(BaseModel):
    """Public description of a selectable strategy (what the UI / MCP list)."""

    name: str
    display_name: str
    description: str
    source: StrategySource
    category: StrategyCategory
    min_candles: int
    default_params: dict[str, Any]
    requires_vision: bool = False
    available: bool = True
    unavailable_reason: str | None = None


# --------------------------------------------------------------------------- helpers


def round_value(value: float | int | None, digits: int = 6) -> float | None:
    """Round to ``digits`` significant figures (prices, oscillators and MACD alike)."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number == 0:
        return 0.0
    return float(f"{number:.{digits}g}")


def series_tail(series: pd.Series, length: int) -> list[float | None]:
    return [round_value(v) for v in series.iloc[-length:].tolist()]


@dataclass
class StrategyContext:
    """Per-run context handed to strategies by the orchestrator."""

    instrument: str = ""
    history_length: int = 10
    prompt_candles: int = 20
    document_prompt_candles: int = 50
    chart_candles: int = 100
    #: Lazily computed, shared standard feature pack for this timeframe (reference strategies).
    feature_pack: Callable[[], dict[str, Any]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class FeatureBuilder:
    """Accumulates indicator values, history, events and a pre-signal."""

    def __init__(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        params: Mapping[str, Any],
        timeframe: str,
        history_length: int = 10,
    ) -> None:
        self.strategy = strategy
        self.df = df
        self.params = dict(params)
        self.timeframe = timeframe
        self.history_length = history_length
        self.latest: dict[str, float | int | str | bool | None] = {}
        self.history: dict[str, list[float | None]] = {}
        self.events: list[str] = []
        self.pre_signal: Signal | None = None
        self.reason: str | None = None
        self.context: dict[str, Any] = {}
        self.images: list[ImageAttachment] = []

    def series(self, name: str, series: pd.Series, *, history: bool = True) -> float | None:
        """Record the latest value (and recent history) of an indicator series."""
        value = round_value(series.iloc[-1]) if len(series) else None
        self.latest[name] = value
        if history:
            self.history[name] = series_tail(series, self.history_length)
        return value

    def value(self, name: str, value: float | int | str | bool | None) -> None:
        if isinstance(value, float):
            value = round_value(value)
        self.latest[name] = value

    def event(self, text: str) -> None:
        if text not in self.events:
            self.events.append(text)

    def signal(self, signal: Signal, reason: str) -> None:
        self.pre_signal = signal
        self.reason = reason

    def build(self) -> StrategyFeatures:
        last_time = self.df.index[-1].to_pydatetime() if len(self.df) else None
        return StrategyFeatures(
            strategy=self.strategy.name,
            display_name=self.strategy.display_name,
            description=self.strategy.description,
            source=self.strategy.source,
            category=self.strategy.category,
            timeframe=self.timeframe,
            candles_available=len(self.df),
            candles_required=self.strategy.required_candles(self.params),
            params=self.params,
            latest=self.latest,
            history=self.history,
            events=self.events,
            pre_signal=self.pre_signal,
            pre_signal_reason=self.reason,
            images=self.images,
            context=self.context,
            last_candle_time=last_time,
        )


def coerce_param(name: str, value: Any, default: Any) -> Any:
    """Coerce a configured parameter to the type of its default."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise ValueError(f"parameter '{name}' must be true/false")
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
            raise ValueError(f"parameter '{name}' must be a whole number")
        return int(value)
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"parameter '{name}' must be a number")
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def resolve_params(
    strategy: Strategy, config: StrategyParamsConfig | None, timeframe: str
) -> dict[str, Any]:
    """Resolution order: timeframe override -> configured params -> class defaults."""
    params = dict(strategy.default_params)
    if config is None:
        return params
    for layer_name, layer in (
        ("params", config.params),
        (f"overrides[{timeframe}]", config.overrides.get(timeframe, {})),
    ):
        for key, value in layer.items():
            if key not in strategy.default_params:
                logger.warning(
                    "strategies.yml: '{}' {} has unknown parameter '{}' (ignored)",
                    strategy.name,
                    layer_name,
                    key,
                )
                continue
            try:
                params[key] = coerce_param(key, value, strategy.default_params[key])
            except ValueError as exc:
                logger.warning("strategies.yml: '{}': {} (ignored)", strategy.name, exc)
    return params


# --------------------------------------------------------------------------- base class


class Strategy(ABC):
    """Abstract, timeframe-agnostic strategy.

    Subclasses implement :meth:`analyze`. :meth:`compute` merges parameters and
    returns an ``insufficient_data`` result (instead of failing) when the
    DataFrame is shorter than :meth:`required_candles`.
    """

    name: str
    display_name: str
    description: str
    source: StrategySource
    category: StrategyCategory
    min_candles: int = 30
    default_params: dict[str, Any] = {}  # noqa: RUF012 - read-only class default
    requires_vision: bool = False

    def required_candles(self, params: Mapping[str, Any]) -> int:
        """Minimum candles for these parameters (override when params drive warm-up)."""
        return self.min_candles

    def merge_params(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(self.default_params)
        merged.update(params or {})
        return merged

    async def compute(
        self,
        df: pd.DataFrame,
        params: Mapping[str, Any] | None,
        timeframe: str,
        *,
        context: StrategyContext | None = None,
    ) -> StrategyFeatures:
        merged = self.merge_params(params)
        needed = self.required_candles(merged)
        if len(df) < needed:
            return self.insufficient(df, merged, timeframe, needed)
        return await self.analyze(df, merged, timeframe, context or StrategyContext())

    @abstractmethod
    async def analyze(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        timeframe: str,
        context: StrategyContext,
    ) -> StrategyFeatures:
        """Compute features for ``df`` (already long enough)."""

    def insufficient(
        self,
        df: pd.DataFrame,
        params: Mapping[str, Any],
        timeframe: str,
        needed: int,
        message: str | None = None,
    ) -> StrategyFeatures:
        return StrategyFeatures(
            strategy=self.name,
            display_name=self.display_name,
            description=self.description,
            source=self.source,
            category=self.category,
            timeframe=timeframe,
            status="insufficient_data",
            message=message
            or f"Needs at least {needed} completed candles, only {len(df)} available.",
            candles_available=len(df),
            candles_required=needed,
            params=dict(params),
            last_candle_time=df.index[-1].to_pydatetime() if len(df) else None,
        )

    def info(self, *, available: bool = True, reason: str | None = None) -> StrategyInfo:
        return StrategyInfo(
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            source=self.source,
            category=self.category,
            min_candles=self.required_candles(self.default_params),
            default_params=dict(self.default_params),
            requires_vision=self.requires_vision,
            available=available,
            unavailable_reason=reason,
        )


class IndicatorStrategy(Strategy):
    """Base for built-in strategies: a synchronous :meth:`evaluate` filling a
    :class:`FeatureBuilder`, executed in a worker thread to keep the event loop free."""

    source: StrategySource = "builtin"

    async def analyze(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
        timeframe: str,
        context: StrategyContext,
    ) -> StrategyFeatures:
        builder = FeatureBuilder(self, df, params, timeframe, context.history_length)
        try:
            await asyncio.to_thread(self.evaluate, df, params, builder)
        except InsufficientDataError as exc:
            return self.insufficient(
                df, params, timeframe, exc.needed or self.required_candles(params), str(exc)
            )
        return builder.build()

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, params: dict[str, Any], fb: FeatureBuilder) -> None:
        """Compute indicators, events and the rule-based pre-signal into ``fb``."""


def require(**values: float | None) -> tuple[float, ...]:
    """Return the values (in order) or raise :class:`InsufficientDataError` if any
    indicator is still warming up (``None``)."""
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise InsufficientDataError(f"Indicators not ready yet: {', '.join(missing)}")
    return tuple(value for value in values.values() if value is not None)
