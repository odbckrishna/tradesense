from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fakes import VALID_DECISION, ScriptedProvider, make_router
from tests.helpers import REPO_ROOT, make_candles, make_docx, make_image, make_pdf
from tradesense.config import LibraryConfig
from tradesense.llm.decision import DecisionEngine
from tradesense.llm.router import LLMRouter
from tradesense.strategies.base import StrategyContext
from tradesense.strategies.reference import (
    CardUpdate,
    LibraryError,
    ReferenceLibrary,
)
from tradesense.strategies.registry import StrategyRegistry

CARD = {
    "name": "Double Bottom Reversal",
    "summary": "Two lows at similar level, then neckline breakout",
    "category": "pattern",
    "buy_conditions": ["two swing lows within 0.2% of each other", "close above neckline"],
    "sell_conditions": [],
    "sideways_conditions": ["no second low formed yet"],
    "required_indicators": ["swing_points", "volume", "fibonacci"],
    "notes": "",
}
CARD_JSON = json.dumps(CARD)
IMAGE_DECISION = json.dumps({**VALID_DECISION, "pattern_match": 0.64})


def build(
    tmp_path: Path,
    text_script: list[str | Exception] | None = None,
    vision_script: list[str | Exception] | None = None,
    **config: object,
) -> tuple[ReferenceLibrary, ScriptedProvider, ScriptedProvider, LLMRouter]:
    text = ScriptedProvider("text_llm", text_script or [CARD_JSON])
    vision = ScriptedProvider(
        "vision_llm", vision_script or [CARD_JSON], capabilities=["text", "vision"]
    )
    router = make_router(
        [text, vision],
        roles={"text": ["text_llm"], "vision": ["vision_llm"], "extraction": ["text_llm"]},
    )
    inbox = tmp_path / "inbox"
    inbox.mkdir(exist_ok=True)
    library = ReferenceLibrary(
        tmp_path / "strategy_library",
        LibraryConfig(max_upload_mb=2, **config),  # type: ignore[arg-type]
        router=router,
        import_roots=[inbox],
        reserved_names=lambda: ["ema_crossover"],
    )
    return library, text, vision, router


def registry_for(
    library: ReferenceLibrary, vision: Callable[[], bool] = lambda: True
) -> StrategyRegistry:
    registry = StrategyRegistry(
        rules_dir=REPO_ROOT / "config" / "rules",
        strategies_config_path=REPO_ROOT / "config" / "strategies.yml",
        reference_loader=library.strategies,
        vision_available=vision,
    )
    library.on_change = registry.reload
    registry.reload()
    return registry


# --------------------------------------------------------------------------- upload + extract


async def test_document_upload_extracts_draft_card(tmp_path: Path) -> None:
    library, text, vision, _ = build(tmp_path)
    result = await library.upload("Double Bottom.md", b"# Double bottom\nBuy above neckline.")
    card = result.card
    assert card.id == "double_bottom_ref"
    assert card.status == "draft"
    assert card.source_type == "document"
    assert card.source_file == "documents/double_bottom.md"
    assert card.name == "Double Bottom Reversal"
    assert card.buy_conditions == CARD["buy_conditions"]
    assert card.required_indicators == ["swing_points", "volume"]
    assert card.warnings == ["Unsupported indicator dropped: fibonacci"]
    assert card.extracted_at is not None
    assert card.reviewed_by_user is False
    assert card.extracted_by == "text_llm (text_llm-model)"
    assert (tmp_path / "strategy_library" / "cards" / "double_bottom_ref.yml").is_file()
    assert len(text.calls) == 1
    assert not vision.calls
    # Cached: reading the card back does not trigger another extraction.
    assert library.get(card.id) == card
    assert len(text.calls) == 1


async def test_image_upload_uses_vision_role(tmp_path: Path) -> None:
    library, text, vision, _ = build(tmp_path)
    card = (await library.upload("double_bottom.png", make_image("PNG"))).card
    assert card.source_type == "image"
    assert card.source_file == "images/double_bottom.png"
    assert len(vision.calls) == 1
    assert vision.calls[0].has_images
    assert not text.calls


async def test_pdf_and_docx_text_is_extracted(tmp_path: Path) -> None:
    library, text, _, _ = build(tmp_path)
    pdf = (await library.upload("plan.pdf", make_pdf(["Buy above the 20 EMA."]))).card
    docx_card = (await library.upload("plan.docx", make_docx(["Sell below VWAP."]))).card
    assert "Buy above the 20 EMA." in library.document_text(pdf)
    assert "Sell below VWAP." in library.document_text(docx_card)
    assert "Buy above the 20 EMA." in text.calls[0].messages[0].text


async def test_duplicate_upload_returns_existing_card(tmp_path: Path) -> None:
    library, text, _, _ = build(tmp_path)
    first = await library.upload("a.md", b"# rules")
    second = await library.upload("b.md", b"# rules")
    assert second.duplicate
    assert second.card.id == first.card.id
    assert len(text.calls) == 1


async def test_extraction_failure_keeps_draft_with_warning(tmp_path: Path) -> None:
    library, _, _, _ = build(tmp_path, text_script=["not json at all"])
    card = (await library.upload("vague.md", b"# some notes")).card
    assert card.status == "draft"
    assert card.warnings
    assert card.warnings[-1].startswith("Extraction failed")
    with pytest.raises(LibraryError, match="before approving"):
        library.approve(card.id)


async def test_long_document_is_chunked_and_summarised(tmp_path: Path) -> None:
    summary = "- buy when RSI < 30"
    library, text, _, _ = build(
        tmp_path,
        text_script=[summary, summary, summary, summary, CARD_JSON],
        max_document_chars=2000,
    )
    body = "\n\n".join(f"Rule {i}: " + "detail " * 60 for i in range(16))
    card = (await library.upload("long.md", body.encode())).card
    summaries = [c for c in text.calls if not c.json_mode]
    final = [c for c in text.calls if c.json_mode]
    assert len(summaries) >= 3
    assert len(final) == 1
    assert "summaries of the trading rules" in final[0].messages[0].text
    assert card.name == "Double Bottom Reversal"


async def test_reextract_on_request(tmp_path: Path) -> None:
    library, text, _, _ = build(
        tmp_path, text_script=[CARD_JSON, json.dumps({**CARD, "name": "V2"})]
    )
    card = (await library.upload("strategy.md", b"# rules")).card
    library.approve(card.id)
    again = await library.extract(card.id)
    assert again.name == "V2"
    assert again.status == "draft"  # must be reviewed again
    assert len(text.calls) == 2


# --------------------------------------------------------------------------- review / approve


async def test_review_edit_approve_disable_delete(tmp_path: Path) -> None:
    library, _, _, _ = build(tmp_path)
    registry = registry_for(library)
    card = (await library.upload("double_bottom.png", make_image("PNG"))).card
    assert card.id not in registry  # drafts are not selectable

    edited = library.update(
        card.id,
        CardUpdate(
            name="My Double Bottom",
            sell_conditions=["close below the second low"],
            required_indicators=["rsi", "Swing Points", "elliott_wave"],
        ),
    )
    assert edited.name == "My Double Bottom"
    assert edited.reviewed_by_user
    assert edited.required_indicators == ["rsi", "swing_points"]
    assert "Unsupported indicator dropped: elliott_wave" in edited.warnings
    assert edited.status == "draft"

    approved = library.approve(card.id)
    assert approved.status == "approved"
    assert card.id in registry  # on_change reloaded the registry
    info = next(i for i in registry.list() if i.name == card.id)
    assert info.source == "image"
    assert info.requires_vision
    assert info.display_name == "My Double Bottom"

    library.disable(card.id)
    assert card.id not in registry

    library.delete(card.id)
    assert not (tmp_path / "strategy_library" / "images" / "double_bottom.png").exists()
    with pytest.raises(LibraryError, match="not found"):
        library.get(card.id)


async def test_card_ids_avoid_existing_strategy_names(tmp_path: Path) -> None:
    library, _, _, _ = build(tmp_path)
    first = (await library.upload("ema_crossover.md", b"# one")).card
    second = (await library.upload("ema crossover.md", b"# two")).card
    assert first.id == "ema_crossover_ref"
    assert second.id == "ema_crossover_2_ref"
    assert second.source_file == "documents/ema_crossover_2.md"


def test_invalid_card_files_are_skipped(tmp_path: Path) -> None:
    library, _, _, _ = build(tmp_path)
    (library.cards_dir / "broken.yml").write_text("id: [unclosed", encoding="utf-8")
    cards, errors = library.load_cards()
    assert cards == []
    assert errors
    assert "broken.yml" in errors[0]


async def test_import_file_is_restricted_to_import_roots(tmp_path: Path) -> None:
    library, _, _, _ = build(tmp_path)
    outside = tmp_path / "secrets.md"
    outside.write_text("# private", encoding="utf-8")
    with pytest.raises(LibraryError, match="outside the allowed import folders") as info:
        await library.import_file(outside)
    assert info.value.status_code == 403
    inside = tmp_path / "inbox" / "rules.md"
    inside.write_text("# rules", encoding="utf-8")
    assert (await library.import_file(inside)).card.status == "draft"
    with pytest.raises(LibraryError, match="not found"):
        await library.import_file(tmp_path / "inbox" / "missing.md")


async def test_preview(tmp_path: Path) -> None:
    library, _, _, _ = build(tmp_path)
    md = (await library.upload("p.md", b"# T\n<script>x</script>", extract=False)).card
    preview = library.preview(md)
    assert preview["kind"] == "markdown"
    assert "&lt;script&gt;" in preview["html"]
    img = (await library.upload("p.png", make_image("PNG"), extract=False)).card
    assert library.preview(img) == {"kind": "image", "url": f"/api/library/{img.id}/file"}


# --------------------------------------------------------------------------- running


async def test_document_strategy_features_and_decision(tmp_path: Path) -> None:
    library, text, _, router = build(tmp_path, text_script=[CARD_JSON, json.dumps(VALID_DECISION)])
    registry = registry_for(library)
    card = (await library.upload("rules.md", b"# Rules\nBuy above the neckline.")).card
    library.approve(card.id)
    strategy = registry.get(card.id)
    df = make_candles(299, "5m")
    features = await strategy.compute(df, {}, "5m", context=StrategyContext(instrument="EUR_USD"))
    assert features.status == "ok"
    assert features.source == "document"
    assert features.context["card"]["buy_conditions"] == CARD["buy_conditions"]
    assert "Buy above the neckline." in features.context["document_text"]
    assert set(features.context["required_indicators"]) == {"swing_points", "volume"}
    assert features.latest["ema20"] is not None
    assert any("RSI14" in e for e in features.events)
    assert features.pre_signal is None
    assert not features.images

    decision = await DecisionEngine(router).decide(features, "EUR_USD", df)
    assert decision.signal == "BUY"
    assert decision.strategy_source == "document"
    prompt = text.calls[-1].messages[0].text
    assert "<untrusted_strategy_card_" in prompt
    assert "<untrusted_document_" in prompt
    assert "Last 50 candles" in prompt


async def test_image_strategy_sends_reference_and_live_chart(tmp_path: Path) -> None:
    library, _, vision, router = build(tmp_path, vision_script=[CARD_JSON, IMAGE_DECISION])
    registry = registry_for(library)
    card = (await library.upload("double_bottom.webp", make_image("WEBP"))).card
    library.approve(card.id)
    df = make_candles(299, "15m")
    features = await registry.get(card.id).compute(
        df, {}, "15m", context=StrategyContext(instrument="EUR_USD", chart_candles=100)
    )
    assert [i.label for i in features.images] == ["reference", "live_chart"]
    assert features.images[0].media_type == "image/png"  # webp normalised for providers
    assert features.images[1].to_bytes().startswith(b"\x89PNG")

    decision = await DecisionEngine(router).decide(features, "EUR_USD", df)
    assert decision.decided_by == "vision_llm"
    assert decision.pattern_match == pytest.approx(0.64)
    request = vision.calls[-1]
    assert len(request.messages[0].images) == 2
    assert "REFERENCE chart:" in request.messages[0].text


async def test_image_strategies_hidden_without_vision(tmp_path: Path) -> None:
    library, _, _, _ = build(tmp_path)
    registry = registry_for(library, vision=lambda: False)
    card = (await library.upload("pattern.png", make_image("PNG"))).card
    library.approve(card.id)
    info = next(i for i in registry.list() if i.name == card.id)
    assert info.available is False
    assert "vision" in (info.unavailable_reason or "")
