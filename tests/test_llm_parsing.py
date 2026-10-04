from __future__ import annotations

import json

import pytest

from tests.fakes import VALID_DECISION, VALID_JSON
from tradesense.llm.schemas import (
    Decision,
    DecisionParseError,
    Probability,
    parse_card_payload,
    parse_decision_payload,
    strip_json_fences,
)


def test_valid_json() -> None:
    payload = parse_decision_payload(VALID_JSON)
    assert payload.signal == "BUY"
    assert payload.probability.buy == pytest.approx(0.62)
    assert payload.confidence == pytest.approx(0.71)
    assert payload.key_levels.support == pytest.approx(1.0841)
    assert payload.key_levels.resistance == pytest.approx(1.0875)
    assert payload.observations == VALID_DECISION["observations"]


@pytest.mark.parametrize(
    "wrapped",
    [
        f"```json\n{VALID_JSON}\n```",
        f"```\n{VALID_JSON}\n```",
        f"  ```JSON\n{VALID_JSON}```  ",
        f"Here is my analysis:\n```json\n{VALID_JSON}\n```\nHope it helps.",
        f"Sure! {VALID_JSON}",
    ],
)
def test_fenced_and_wrapped_json(wrapped: str) -> None:
    assert parse_decision_payload(wrapped).signal == "BUY"


def test_strip_json_fences_leaves_plain_text() -> None:
    assert strip_json_fences(VALID_JSON) == VALID_JSON


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I think the market will go up.",
        "{not json at all",
        "[1, 2, 3]",
        json.dumps({**VALID_DECISION, "signal": "HOLD"}),
        json.dumps({k: v for k, v in VALID_DECISION.items() if k != "probability"}),
        json.dumps({**VALID_DECISION, "probability": {"buy": 0.5, "sell": 0.5}}),
        json.dumps({**VALID_DECISION, "probability": {"buy": "lots", "sell": 0, "sideways": 0}}),
        json.dumps({k: v for k, v in VALID_DECISION.items() if k != "confidence"}),
        json.dumps({**VALID_DECISION, "confidence": 250}),
        json.dumps({**VALID_DECISION, "reasoning": ["not", "a", "string"]}),
    ],
)
def test_invalid_outputs_rejected(text: str) -> None:
    with pytest.raises(DecisionParseError):
        parse_decision_payload(text)


def test_probabilities_are_normalised_and_signal_is_argmax() -> None:
    raw = {**VALID_DECISION, "signal": "SELL", "probability": {"buy": 3, "sell": 1, "sideways": 1}}
    payload = parse_decision_payload(json.dumps(raw))
    total = payload.probability.buy + payload.probability.sell + payload.probability.sideways
    assert total == pytest.approx(1.0, abs=1e-3)
    assert payload.signal == "BUY"  # the stated SELL contradicts the probabilities


def test_percentages_accepted() -> None:
    raw = {
        **VALID_DECISION,
        "probability": {"buy": 20, "sell": 70, "sideways": 10},
        "confidence": "80%",
        "pattern_match": 55,
    }
    payload = parse_decision_payload(json.dumps(raw))
    assert payload.signal == "SELL"
    assert payload.probability.sell == pytest.approx(0.7)
    assert payload.confidence == pytest.approx(0.8)
    assert payload.pattern_match == pytest.approx(0.55)


def test_lists_and_strings_are_bounded() -> None:
    raw = {**VALID_DECISION, "observations": ["x" * 1000] * 50, "reasoning": "y" * 10_000}
    payload = parse_decision_payload(json.dumps(raw))
    assert len(payload.observations) == 8
    assert len(payload.observations[0]) == 300
    assert len(payload.reasoning) == 1500


def test_bad_key_levels_become_null() -> None:
    raw = {**VALID_DECISION, "key_levels": {"support": "around 1.08", "resistance": -5}}
    payload = parse_decision_payload(json.dumps(raw))
    assert payload.key_levels.support is None
    assert payload.key_levels.resistance is None


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.6, 0.2, 0.2), "BUY"),
        ((0.2, 0.6, 0.2), "SELL"),
        ((0.4, 0.4, 0.2), "SIDEWAYS"),  # BUY == SELL is conflict
        ((0.4, 0.2, 0.4), "SIDEWAYS"),  # tie with SIDEWAYS
        ((0.0, 0.0, 0.0), "SIDEWAYS"),
    ],
)
def test_probability_top(values: tuple[float, float, float], expected: str) -> None:
    assert Probability.normalized(*values).top() == expected


def test_fallback_decision_is_sideways_with_zero_confidence() -> None:
    decision = Decision.fallback("builtin", "all providers failed")
    assert decision.signal == "SIDEWAYS"
    assert decision.confidence == 0
    assert decision.observations == ["LLM parse error"]
    assert decision.probability.sideways == 1.0


def test_card_payload_parsing() -> None:
    card = parse_card_payload(
        json.dumps(
            {
                "name": "Double Bottom",
                "summary": "Two lows then neckline break",
                "category": "pattern",
                "buy_conditions": ["close above neckline"],
                "sell_conditions": [],
                "sideways_conditions": "no second low yet",
                "required_indicators": ["swing_points", "volume"],
                "notes": None,
            }
        )
    )
    assert card.sideways_conditions == ["no second low yet"]
    assert card.notes == ""
    with pytest.raises(DecisionParseError):
        parse_card_payload(json.dumps({"name": "empty"}))
