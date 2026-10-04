from __future__ import annotations

import asyncio

import pytest

from tests.fakes import VALID_JSON, ScriptedProvider, make_router
from tests.helpers import make_candles
from tradesense.config import LLMSettings
from tradesense.llm.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMQuotaError,
    LLMRateLimitError,
    LLMTransientError,
    classify_status,
)
from tradesense.llm.decision import DecisionEngine
from tradesense.llm.messages import ImagePart, LLMRequest, Message
from tradesense.llm.router import AllProvidersFailedError, NoProviderError
from tradesense.llm.schemas import parse_decision_payload
from tradesense.strategies.base import ImageAttachment, StrategyFeatures

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def text_request() -> LLMRequest:
    return LLMRequest(system="sys", messages=[Message.user("hello")])


def image_request() -> LLMRequest:
    return LLMRequest(
        system="sys",
        messages=[Message.user("look", ImagePart(data=PNG, media_type="image/png"))],
    )


def features(images: bool = False) -> StrategyFeatures:
    return StrategyFeatures(
        strategy="ema_crossover",
        display_name="EMA Crossover",
        description="test",
        source="image" if images else "builtin",
        category="trend",
        timeframe="5m",
        candles_available=300,
        candles_required=30,
        latest={"ema_fast": 1.1},
        events=["Bullish crossover 2 candles ago"],
        pre_signal="BUY",
        pre_signal_reason="test",
        images=[ImageAttachment.from_bytes("reference", PNG, "image/png")] if images else [],
    )


# --------------------------------------------------------------------------- fallbacks


async def test_rate_limit_retries_then_falls_back() -> None:
    primary = ScriptedProvider("primary", [LLMRateLimitError("429 slow down")])
    backup = ScriptedProvider("backup", [VALID_JSON])
    router = make_router([primary, backup])

    response = await router.generate("text", text_request())

    assert response.provider == "backup"
    assert len(primary.calls) == 3  # 1 + settings.retries
    assert len(backup.calls) == 1
    assert router.role_status()["text"]["fallback_in_use"] is True
    assert router.role_status()["text"]["active"] == "backup"


async def test_transient_error_recovers_on_retry() -> None:
    primary = ScriptedProvider("primary", [LLMTransientError("503"), VALID_JSON])
    backup = ScriptedProvider("backup", [VALID_JSON])
    router = make_router([primary, backup])
    response = await router.generate("text", text_request())
    assert response.provider == "primary"
    assert len(primary.calls) == 2
    assert not backup.calls


async def test_auth_error_skips_retries_and_provider() -> None:
    primary = ScriptedProvider("primary", [LLMAuthError("401 invalid key")])
    backup = ScriptedProvider("backup", [VALID_JSON])
    router = make_router([primary, backup])

    first = await router.generate("text", text_request())
    assert first.provider == "backup"
    assert len(primary.calls) == 1  # no retries on auth errors
    assert router.state["primary"].disabled

    await router.generate("text", text_request())
    assert len(primary.calls) == 1  # rejected key is not tried again until reload


async def test_exhausted_quota_is_not_retried() -> None:
    error = classify_status(
        429, "Error code: 429 - {'type': 'insufficient_quota', 'code': 'credit_balance_exhausted'}"
    )
    assert isinstance(error, LLMQuotaError)
    assert isinstance(classify_status(429, "Rate limit reached, retry in 2s"), LLMRateLimitError)
    primary = ScriptedProvider("primary", [error])
    backup = ScriptedProvider("backup", [VALID_JSON])
    router = make_router([primary, backup])
    assert (await router.generate("text", text_request())).provider == "backup"
    assert len(primary.calls) == 1
    assert router.state["primary"].disabled
    assert router.role_status()["text"]["active"] == "backup"


async def test_bad_request_falls_back_without_retry() -> None:
    primary = ScriptedProvider("primary", [LLMBadRequestError("400")])
    backup = ScriptedProvider("backup", [VALID_JSON])
    router = make_router([primary, backup])
    assert (await router.generate("text", text_request())).provider == "backup"
    assert len(primary.calls) == 1


async def test_image_request_skips_text_only_providers() -> None:
    text_only = ScriptedProvider("text_only", [VALID_JSON], capabilities=["text"])
    vision = ScriptedProvider("vision", [VALID_JSON], capabilities=["text", "vision"])
    router = make_router([text_only, vision])

    response = await router.generate("text", image_request())
    assert response.provider == "vision"
    assert not text_only.calls
    assert router.vision_available() is False  # no 'vision' role configured here


async def test_image_request_without_vision_provider() -> None:
    router = make_router([ScriptedProvider("text_only", [VALID_JSON])])
    with pytest.raises(NoProviderError, match="vision-capable"):
        await router.generate("text", image_request())


async def test_all_providers_fail_raises() -> None:
    router = make_router(
        [
            ScriptedProvider("a", [LLMTransientError("down")]),
            ScriptedProvider("b", [LLMAuthError("401")]),
        ]
    )
    with pytest.raises(AllProvidersFailedError) as info:
        await router.generate("text", text_request())
    assert len(info.value.errors) == 2


async def test_unexpected_adapter_exception_is_contained() -> None:
    router = make_router(
        [
            ScriptedProvider("buggy", [ZeroDivisionError("oops")]),
            ScriptedProvider("ok", [VALID_JSON]),
        ]
    )
    assert (await router.generate("text", text_request())).provider == "ok"


async def test_timeout_is_transient() -> None:
    settings = LLMSettings(retries=0, retry_backoff_seconds=0, timeout_seconds=0.05)
    slow = ScriptedProvider("slow", [VALID_JSON], delay=1.0, settings=settings)
    fast = ScriptedProvider("fast", [VALID_JSON], settings=settings)
    router = make_router([slow, fast], settings=settings)
    assert (await router.generate("text", text_request())).provider == "fast"
    assert "no response within" in (router.state["slow"].error or "")


# --------------------------------------------------------------------------- structured output


async def test_invalid_json_gets_one_correction_then_succeeds() -> None:
    primary = ScriptedProvider("primary", ["Sure, I think BUY.", VALID_JSON])
    router = make_router([primary])
    result = await router.generate_structured("text", text_request(), parse_decision_payload)
    assert result.value.signal == "BUY"
    assert len(primary.calls) == 2
    correction = primary.calls[1].messages
    assert correction[1].role == "assistant"
    assert "could not be used" in correction[2].text
    assert len(result.responses) == 2


async def test_invalid_json_twice_moves_to_next_provider() -> None:
    primary = ScriptedProvider("primary", ["nope"])
    backup = ScriptedProvider("backup", [VALID_JSON])
    router = make_router([primary, backup])
    result = await router.generate_structured("text", text_request(), parse_decision_payload)
    assert result.provider == "backup"
    assert result.fallback_used
    assert len(primary.calls) == 2


async def test_all_fail_to_parse_gives_sideways_decision() -> None:
    router = make_router([ScriptedProvider("a", ["garbage"]), ScriptedProvider("b", ["{bad"])])
    decision = await DecisionEngine(router).decide(features(), "EUR_USD", make_candles(50))
    assert decision.signal == "SIDEWAYS"
    assert decision.confidence == 0
    assert "LLM parse error" in decision.observations
    assert decision.llm_calls == 4
    assert decision.input_tokens == 400
    assert decision.cost_usd == pytest.approx(4 * (100 * 1 + 50 * 2) / 1e6)


async def test_all_providers_down_gives_sideways_decision() -> None:
    router = make_router([ScriptedProvider("a", [LLMTransientError("down")])])
    decision = await DecisionEngine(router).decide(features(), "EUR_USD", make_candles(50))
    assert decision.signal == "SIDEWAYS"
    assert decision.observations == ["LLM unavailable"]
    assert decision.decided_by == "none"


async def test_decision_engine_success_and_vision_role() -> None:
    text = ScriptedProvider("text", [VALID_JSON])
    vision = ScriptedProvider("vision", [VALID_JSON], capabilities=["text", "vision"])
    router = make_router([text, vision], roles={"text": ["text"], "vision": ["vision"]})
    engine = DecisionEngine(router)

    decision = await engine.decide(features(), "EUR_USD", make_candles(50))
    assert decision.decided_by == "text"
    assert decision.model == "text-model"
    assert decision.strategy_source == "builtin"
    assert decision.pattern_match is None
    assert decision.signal == "BUY"

    image_decision = await engine.decide(features(images=True), "EUR_USD", make_candles(50))
    assert image_decision.decided_by == "vision"
    assert vision.calls[0].has_images
    assert router.vision_available()


# --------------------------------------------------------------------------- limits / status


async def test_global_concurrency_cap() -> None:
    settings = LLMSettings(max_concurrency=2, retry_backoff_seconds=0)
    provider = ScriptedProvider("p", [VALID_JSON], delay=0.05, settings=settings)
    router = make_router([provider], settings=settings)
    await asyncio.gather(*(router.generate("text", text_request()) for _ in range(6)))
    assert provider.peak == 2


async def test_per_provider_concurrency_cap() -> None:
    settings = LLMSettings(max_concurrency=8, retry_backoff_seconds=0)
    provider = ScriptedProvider("p", [VALID_JSON], delay=0.05, settings=settings, max_concurrency=1)
    router = make_router([provider], settings=settings)
    await asyncio.gather(*(router.generate("text", text_request()) for _ in range(4)))
    assert provider.peak == 1


async def test_health_check_and_status() -> None:
    good = ScriptedProvider("good", [VALID_JSON])
    bad = ScriptedProvider("bad", [VALID_JSON], healthy=False)
    router = make_router([bad, good])
    result = await router.check_health()
    assert result == {"bad": False, "good": True}
    status = router.role_status()["text"]
    assert status["primary"] == "bad"
    assert status["active"] == "good"
    assert status["fallback_in_use"] is True
    providers = router.provider_status()
    assert providers["bad"]["healthy"] is False
    assert providers["good"]["model"] == "good-model"


async def test_usage_is_tracked() -> None:
    router = make_router([ScriptedProvider("p", [VALID_JSON])])
    await router.generate("text", text_request())
    snapshot = router.usage.snapshot()
    assert snapshot["total_calls"] == 1
    assert snapshot["total_tokens"] == 150
    assert snapshot["providers"]["p"]["cost_usd"] == pytest.approx(0.0002)


async def test_decision_only_provider_serves_decisions_only() -> None:
    jev = ScriptedProvider("jev", [VALID_JSON], capabilities=["decision"])
    llm = ScriptedProvider("llm", [VALID_JSON])
    router = make_router([jev, llm], roles={"text": ["jev", "llm"], "extraction": ["jev", "llm"]})
    decision = LLMRequest(system="sys", messages=[Message.user("market data")], task="decision")
    assert (await router.generate("text", decision)).provider == "jev"
    assert (await router.generate("text", text_request())).provider == "llm"  # e.g. summaries
    assert [p.name for p in router.chain("extraction")] == ["llm"]
    assert router.role_status()["extraction"]["active"] == "llm"
    assert [r.task for r in jev.calls] == ["decision"]
