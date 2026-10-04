from __future__ import annotations

import pytest

from tradesense.engine.orchestrator import compute_consensus
from tradesense.llm.schemas import Decision, KeyLevels, Probability


def decision(
    buy: float,
    sell: float,
    sideways: float,
    confidence: float,
    *,
    support: float | None = None,
    resistance: float | None = None,
    decided_by: str = "llm",
) -> Decision:
    probability = Probability.normalized(buy, sell, sideways)
    return Decision(
        signal=probability.top(),
        probability=probability,
        confidence=confidence,
        key_levels=KeyLevels(support=support, resistance=resistance),
        strategy_source="builtin",
        decided_by=decided_by,
        model="m",
    )


def test_unanimous() -> None:
    result = compute_consensus(
        [
            ("a", decision(0.8, 0.1, 0.1, 0.8)),
            ("b", decision(0.7, 0.1, 0.2, 0.6)),
        ]
    )
    assert result is not None
    assert result.signal == "BUY"
    assert result.confidence == pytest.approx(0.7)  # mean confidence x full agreement
    assert result.strategy_source == "consensus"
    assert result.observations[0].startswith("2 BUY / 0 SELL / 0 SIDEWAYS")


def test_confidence_weighted_average() -> None:
    result = compute_consensus(
        [
            ("confident_buy", decision(0.8, 0.1, 0.1, 0.9)),
            ("timid_sell", decision(0.1, 0.8, 0.1, 0.1)),
        ]
    )
    assert result is not None
    # p_buy = (0.9*0.8 + 0.1*0.1) / 1.0 = 0.73 ; p_sell = 0.17 ; p_sideways = 0.10
    assert result.probability.buy == pytest.approx(0.73)
    assert result.probability.sell == pytest.approx(0.17)
    assert result.probability.sideways == pytest.approx(0.10)
    assert result.signal == "BUY"
    # mean confidence 0.5 x agreement 0.9
    assert result.confidence == pytest.approx(0.45)
    assert "confident_buy" in result.observations[2]


def test_disagreement_lowers_confidence() -> None:
    result = compute_consensus(
        [
            ("a", decision(0.7, 0.2, 0.1, 0.6)),
            ("b", decision(0.2, 0.7, 0.1, 0.6)),
        ]
    )
    assert result is not None
    assert result.signal == "SIDEWAYS"  # BUY and SELL tie: conflicting evidence
    assert result.confidence == pytest.approx(0.0)


def test_zero_confidence_is_sideways() -> None:
    result = compute_consensus([("a", decision(0.9, 0.05, 0.05, 0.0))])
    assert result is not None
    assert result.signal == "SIDEWAYS"
    assert result.confidence == 0


def test_fallback_decisions_are_excluded() -> None:
    fallback = Decision.fallback("builtin", "LLM down")
    assert compute_consensus([("x", fallback)]) is None
    result = compute_consensus([("x", fallback), ("a", decision(0.1, 0.8, 0.1, 0.5))])
    assert result is not None
    assert result.signal == "SELL"
    assert any("unavailable" in o for o in result.observations)


def test_key_levels_use_median() -> None:
    result = compute_consensus(
        [
            ("a", decision(0.6, 0.2, 0.2, 0.5, support=1.0, resistance=1.2)),
            ("b", decision(0.6, 0.2, 0.2, 0.5, support=1.1, resistance=1.3)),
            ("c", decision(0.6, 0.2, 0.2, 0.5, support=5.0)),
        ]
    )
    assert result is not None
    assert result.key_levels.support == pytest.approx(1.1)
    assert result.key_levels.resistance == pytest.approx(1.25)


def test_empty() -> None:
    assert compute_consensus([]) is None
