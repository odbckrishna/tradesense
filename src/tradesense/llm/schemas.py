"""Decision and StrategyCard models, plus parsing of LLM JSON output.

The LLM returns the *analysis* fields of a :class:`Decision`; TradeSense adds the
bookkeeping fields (``strategy_source``, ``decided_by``, ``model``, latency, cost).
Parsing strips ```json fences, accepts 0-100 percentages, renormalises
probabilities and forces ``signal`` to the highest-probability outcome. Output that
does not match the schema raises :class:`DecisionParseError`.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from tradesense.indicators.technical import SUPPORTED_INDICATORS
from tradesense.strategies.base import CATEGORIES, StrategyCategory

Signal = Literal["BUY", "SELL", "SIDEWAYS"]
DecisionSource = Literal["builtin", "rule", "document", "image", "consensus"]
SIGNALS: tuple[Signal, ...] = ("BUY", "SELL", "SIDEWAYS")

MAX_OBSERVATIONS = 8
MAX_OBSERVATION_CHARS = 300
MAX_REASONING_CHARS = 1500


class DecisionParseError(ValueError):
    """The model's output is not valid JSON matching the decision schema."""


# --------------------------------------------------------------------------- decision


class Probability(BaseModel):
    buy: float = Field(ge=0, le=1)
    sell: float = Field(ge=0, le=1)
    sideways: float = Field(ge=0, le=1)

    @classmethod
    def normalized(cls, buy: float, sell: float, sideways: float) -> Probability:
        values = [max(0.0, v) if math.isfinite(v) else 0.0 for v in (buy, sell, sideways)]
        total = sum(values)
        if total <= 0:
            return cls(buy=0.0, sell=0.0, sideways=1.0)
        b, s, w = (round(v / total, 4) for v in values)
        return cls(buy=b, sell=s, sideways=w)

    def top(self) -> Signal:
        """Highest-probability outcome. Ties involving SIDEWAYS, or BUY == SELL, are
        SIDEWAYS (conflicting evidence is not a directional call)."""
        best = max(self.buy, self.sell, self.sideways)
        if math.isclose(self.sideways, best, abs_tol=1e-9):
            return "SIDEWAYS"
        if math.isclose(self.buy, self.sell, abs_tol=1e-9):
            return "SIDEWAYS"
        return "BUY" if self.buy > self.sell else "SELL"

    def as_dict(self) -> dict[str, float]:
        return {"buy": self.buy, "sell": self.sell, "sideways": self.sideways}


class KeyLevels(BaseModel):
    support: float | None = None
    resistance: float | None = None


class DecisionPayload(BaseModel):
    """The part of a decision produced by the LLM (validated)."""

    signal: Signal
    probability: Probability
    confidence: float = Field(ge=0, le=1)
    observations: list[str] = Field(default_factory=list)
    reasoning: str = ""
    key_levels: KeyLevels = Field(default_factory=KeyLevels)
    pattern_match: float | None = Field(default=None, ge=0, le=1)


class Decision(BaseModel):
    """A BUY / SELL / SIDEWAYS decision for one strategy (or a consensus)."""

    signal: Signal
    probability: Probability
    confidence: float = Field(ge=0, le=1)
    observations: list[str] = Field(default_factory=list)
    reasoning: str = ""
    key_levels: KeyLevels = Field(default_factory=KeyLevels)
    pattern_match: float | None = Field(default=None, ge=0, le=1)
    strategy_source: DecisionSource
    decided_by: str
    model: str
    latency_ms: int = 0
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    fallback_used: bool = False
    error: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def fallback(
        cls,
        source: DecisionSource,
        reason: str,
        observation: str = "LLM parse error",
        **extra: Any,
    ) -> Decision:
        """SIDEWAYS with confidence 0: used when no provider produced a valid decision."""
        return cls(
            signal="SIDEWAYS",
            probability=Probability(buy=0.0, sell=0.0, sideways=1.0),
            confidence=0.0,
            observations=[observation],
            reasoning=f"No valid LLM decision: {reason}"[:MAX_REASONING_CHARS],
            strategy_source=source,
            decided_by="none",
            model="none",
            error=reason[:500],
            **extra,
        )


# --------------------------------------------------------------------------- parsing

_FENCED = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", re.DOTALL)
_EMBEDDED_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)


def strip_json_fences(text: str) -> str:
    """Remove a surrounding ```json ... ``` fence (or extract the first fenced block)."""
    stripped = text.strip()
    match = _FENCED.match(stripped)
    if match:
        return match.group(1).strip()
    embedded = _EMBEDDED_FENCE.search(stripped)
    if embedded:
        return embedded.group(1).strip()
    return stripped


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the JSON object in ``text`` (fences stripped). Raises DecisionParseError."""
    if not text or not text.strip():
        raise DecisionParseError("empty response")
    cleaned = strip_json_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise DecisionParseError(f"no JSON object found: {first_error.msg}") from None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise DecisionParseError(f"invalid JSON ({exc.msg} at position {exc.pos})") from None
    if not isinstance(data, dict):
        raise DecisionParseError("expected a JSON object at the top level")
    return data


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise DecisionParseError(f"'{field}' must be a number")
    try:
        number = float(str(value).strip().rstrip("%")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise DecisionParseError(f"'{field}' must be a number") from None
    if not math.isfinite(number) or number < 0:
        raise DecisionParseError(f"'{field}' must be a finite, non-negative number")
    return number


def _unit(value: Any, field: str) -> float:
    """0-1 value; 0-100 percentages are accepted and scaled."""
    number = _number(value, field)
    if number > 1:
        if number <= 100:
            return number / 100
        raise DecisionParseError(f"'{field}' must be between 0 and 1")
    return number


def _optional_price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def parse_decision_payload(text: str) -> DecisionPayload:
    """Validate LLM output against the decision schema (see module docstring)."""
    data = {str(k).strip().lower(): v for k, v in extract_json_object(text).items()}

    raw_signal = data.get("signal")
    if not isinstance(raw_signal, str) or raw_signal.strip().upper() not in SIGNALS:
        raise DecisionParseError("'signal' must be one of BUY, SELL, SIDEWAYS")

    raw_prob = data.get("probability", data.get("probabilities"))
    if not isinstance(raw_prob, dict):
        raise DecisionParseError("'probability' must be an object with buy, sell, sideways")
    prob = {str(k).strip().lower(): v for k, v in raw_prob.items()}
    missing = [k for k in ("buy", "sell", "sideways") if k not in prob]
    if missing:
        raise DecisionParseError(f"'probability' is missing {', '.join(missing)}")
    values = [_number(prob[k], f"probability.{k}") for k in ("buy", "sell", "sideways")]
    if sum(values) > 1.5:  # percentages
        values = [v / 100 for v in values]
    probability = Probability.normalized(*values)

    if "confidence" not in data:
        raise DecisionParseError("'confidence' is required")
    confidence = _unit(data["confidence"], "confidence")

    raw_obs = data.get("observations", [])
    if isinstance(raw_obs, str):
        raw_obs = [raw_obs]
    if not isinstance(raw_obs, list):
        raise DecisionParseError("'observations' must be a list of strings")
    observations = [
        str(o).strip()[:MAX_OBSERVATION_CHARS]
        for o in raw_obs[:MAX_OBSERVATIONS]
        if isinstance(o, str | int | float) and str(o).strip()
    ]

    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise DecisionParseError("'reasoning' must be a string")

    levels = data.get("key_levels") or {}
    if not isinstance(levels, dict):
        levels = {}
    pattern = data.get("pattern_match")
    pattern_match = None if pattern is None else _unit(pattern, "pattern_match")

    return DecisionPayload(
        signal=probability.top(),  # signal is always the highest-probability outcome
        probability=probability,
        confidence=round(confidence, 4),
        observations=observations,
        reasoning=reasoning.strip()[:MAX_REASONING_CHARS],
        key_levels=KeyLevels(
            support=_optional_price(levels.get("support")),
            resistance=_optional_price(levels.get("resistance")),
        ),
        pattern_match=None if pattern_match is None else round(pattern_match, 4),
    )


#: JSON schema sent to providers with native structured output.
DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": list(SIGNALS)},
        "probability": {
            "type": "object",
            "properties": {
                "buy": {"type": "number"},
                "sell": {"type": "number"},
                "sideways": {"type": "number"},
            },
            "required": ["buy", "sell", "sideways"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number"},
        "observations": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "key_levels": {
            "type": "object",
            "properties": {
                "support": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "resistance": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
            "required": ["support", "resistance"],
            "additionalProperties": False,
        },
        "pattern_match": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    },
    "required": [
        "signal",
        "probability",
        "confidence",
        "observations",
        "reasoning",
        "key_levels",
        "pattern_match",
    ],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- strategy cards

CardStatus = Literal["draft", "approved", "disabled"]
CardSourceType = Literal["document", "image"]

MAX_CONDITIONS = 20
MAX_CONDITION_CHARS = 500


def _clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("must be a list of strings")
    cleaned = [str(v).strip()[:MAX_CONDITION_CHARS] for v in values if str(v).strip()]
    return cleaned[:MAX_CONDITIONS]


class StrategyCard(BaseModel):
    """Reviewed description of an uploaded reference strategy (document or image)."""

    id: str
    name: str
    source_type: CardSourceType
    source_file: str  # relative to strategy_library/, e.g. images/double_bottom.png
    status: CardStatus = "draft"
    summary: str = ""
    category: StrategyCategory = "pattern"
    buy_conditions: list[str] = Field(default_factory=list)
    sell_conditions: list[str] = Field(default_factory=list)
    sideways_conditions: list[str] = Field(default_factory=list)
    required_indicators: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""
    extracted_at: datetime | None = None
    reviewed_by_user: bool = False
    # provenance
    original_filename: str = ""
    media_type: str = ""
    sha256: str = ""
    size_bytes: int = 0
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None
    extracted_by: str | None = None

    @field_validator("buy_conditions", "sell_conditions", "sideways_conditions", mode="before")
    @classmethod
    def _conditions(cls, value: Any) -> list[str]:
        return _clean_list(value)

    @field_validator("warnings", mode="before")
    @classmethod
    def _warnings(cls, value: Any) -> list[str]:
        return _clean_list(value)

    @field_validator("summary")
    @classmethod
    def _summary(cls, value: str) -> str:
        return value.strip()[:2000]

    @field_validator("notes")
    @classmethod
    def _notes(cls, value: str) -> str:
        return value.strip()[:5000]

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()[:120]
        if not value:
            raise ValueError("name must not be empty")
        return value

    @field_validator("category", mode="before")
    @classmethod
    def _category(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower().replace(" ", "_").replace("-", "_")
            return value if value in CATEGORIES else "pattern"
        return value

    def with_indicators(self, indicators: Any) -> StrategyCard:
        """Keep only supported indicators; unknown ones are dropped and listed in warnings."""
        known, unknown = split_indicators(indicators)
        warnings = [w for w in self.warnings if not w.startswith("Unsupported indicator")]
        warnings += [f"Unsupported indicator dropped: {name}" for name in unknown]
        return self.model_copy(update={"required_indicators": known, "warnings": warnings})


def split_indicators(values: Any) -> tuple[list[str], list[str]]:
    """Split names into (supported, unsupported). Matching is case-insensitive."""
    if values is None:
        return [], []
    if isinstance(values, str):
        values = [values]
    known: list[str] = []
    unknown: list[str] = []
    for raw in values if isinstance(values, list) else []:
        name = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
        if not name:
            continue
        target = known if name in SUPPORTED_INDICATORS else unknown
        if name not in target:
            target.append(name)
    return known, unknown


class CardExtractionPayload(BaseModel):
    """What the extraction LLM returns for a document or image."""

    name: str = ""
    summary: str = ""
    category: str = "pattern"
    buy_conditions: list[str] = Field(default_factory=list)
    sell_conditions: list[str] = Field(default_factory=list)
    sideways_conditions: list[str] = Field(default_factory=list)
    required_indicators: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator(
        "buy_conditions",
        "sell_conditions",
        "sideways_conditions",
        "required_indicators",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: Any) -> list[str]:
        return _clean_list(value)


def parse_card_payload(text: str) -> CardExtractionPayload:
    data = {str(k).strip().lower(): v for k, v in extract_json_object(text).items()}
    for key in ("name", "summary", "notes", "category"):
        if key in data and not isinstance(data[key], str):
            data[key] = "" if data[key] is None else str(data[key])
    try:
        payload = CardExtractionPayload.model_validate(data)
    except ValueError as exc:
        raise DecisionParseError(f"card does not match the schema: {exc}") from None
    if not (payload.summary or payload.buy_conditions or payload.sell_conditions):
        raise DecisionParseError("card has no summary or conditions")
    return payload


CARD_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "summary": {"type": "string"},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "buy_conditions": {"type": "array", "items": {"type": "string"}},
        "sell_conditions": {"type": "array", "items": {"type": "string"}},
        "sideways_conditions": {"type": "array", "items": {"type": "string"}},
        "required_indicators": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": [
        "name",
        "summary",
        "category",
        "buy_conditions",
        "sell_conditions",
        "sideways_conditions",
        "required_indicators",
        "notes",
    ],
    "additionalProperties": False,
}
