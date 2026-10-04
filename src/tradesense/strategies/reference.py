"""Reference strategies: uploaded documents and reference chart images.

Lifecycle: upload -> extract -> review -> approve -> run.

* UPLOAD: validated by signature, size-limited, filename sanitised, stored under
  ``strategy_library/documents`` or ``strategy_library/images``.
* EXTRACT (once): documents go to the ``extraction`` role (chunked and summarised
  when long), images to the ``vision`` role; the result is a draft STRATEGY CARD in
  ``strategy_library/cards``. Extractions are cached; re-extract only on request.
* REVIEW / APPROVE: only approved cards become selectable strategies.
* RUN: :class:`DocumentStrategy` / :class:`ImageStrategy` on any timeframe.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from tradesense.config import LibraryConfig
from tradesense.llm.base import LLMError
from tradesense.llm.messages import ImagePart, LLMRequest
from tradesense.llm.prompts import (
    build_chunk_summary_request,
    build_document_extraction_request,
    build_image_extraction_request,
)
from tradesense.llm.router import LLMRouter
from tradesense.llm.schemas import StrategyCard, parse_card_payload
from tradesense.strategies.base import (
    FeatureBuilder,
    ImageAttachment,
    Strategy,
    StrategyCategory,
    StrategyContext,
    StrategyFeatures,
    StrategySource,
)
from tradesense.strategies.chart_renderer import render_chart_async
from tradesense.strategies.feature_pack import compute_feature_pack, compute_required_indicators
from tradesense.strategies.ingestion import (
    EXTENSIONS,
    KINDS,
    UploadError,
    chunk_text,
    extract_text,
    prepare_image_for_llm,
    render_markdown,
    validate_upload,
)

CARD_ID = re.compile(r"^[a-z0-9_]{1,100}$")
MAX_CHUNKS = 10
PREVIEW_CHARS = 200_000


class LibraryError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class CardNotFoundError(LibraryError):
    def __init__(self, card_id: str) -> None:
        super().__init__(f"Reference strategy '{card_id}' not found", 404)


class CardUpdate(BaseModel):
    """Fields a user may edit during review (status changes use approve/disable)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    summary: str | None = None
    category: StrategyCategory | None = None
    buy_conditions: list[str] | None = None
    sell_conditions: list[str] | None = None
    sideways_conditions: list[str] | None = None
    required_indicators: list[str] | None = None
    notes: str | None = None


@dataclass
class UploadResult:
    card: StrategyCard
    duplicate: bool = False


def _now() -> datetime:
    return datetime.now(UTC)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:60] or "strategy"


class ReferenceLibrary:
    """Storage and lifecycle of reference strategy files and their cards."""

    def __init__(
        self,
        root: Path,
        config: LibraryConfig,
        *,
        router: LLMRouter | None = None,
        import_roots: Sequence[Path] = (),
        reserved_names: Callable[[], Iterable[str]] | None = None,
        on_change: Callable[[], object] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.config = config
        self.router = router
        self.import_roots = [Path(p).expanduser().resolve() for p in import_roots]
        self.reserved_names = reserved_names or (lambda: ())
        self.on_change = on_change
        self.documents_dir = self.root / "documents"
        self.images_dir = self.root / "images"
        self.cards_dir = self.root / "cards"
        self.cache_dir = self.root / ".cache"
        for directory in (self.documents_dir, self.images_dir, self.cards_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._image_cache: dict[str, tuple[float, bytes, str]] = {}

    # ------------------------------------------------------------------ paths

    def card_path(self, card_id: str) -> Path:
        if not CARD_ID.match(card_id):
            raise CardNotFoundError(card_id)
        return self.cards_dir / f"{card_id}.yml"

    def source_path(self, card: StrategyCard) -> Path:
        path = (self.root / card.source_file).resolve()
        if not path.is_relative_to(self.root):
            raise LibraryError(f"Card '{card.id}' points outside the library", 400)
        return path

    def _text_cache(self, card_id: str) -> Path:
        return self.cache_dir / f"{card_id}.txt"

    # ------------------------------------------------------------------ cards

    def load_cards(self) -> tuple[list[StrategyCard], list[str]]:
        cards: list[StrategyCard] = []
        errors: list[str] = []
        for path in sorted(self.cards_dir.glob("*.yml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                card = StrategyCard.model_validate(data)
                if card.id != path.stem:
                    raise ValueError(f"id '{card.id}' does not match the file name")
                cards.append(card)
            except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
                errors.append(f"card {path.name}: {str(exc).splitlines()[0][:200]}")
        return cards, errors

    def list_cards(self) -> list[StrategyCard]:
        cards, errors = self.load_cards()
        for error in errors:
            logger.warning("Skipping invalid strategy card: {}", error)
        return sorted(cards, key=lambda c: c.uploaded_at, reverse=True)

    def get(self, card_id: str) -> StrategyCard:
        path = self.card_path(card_id)
        if not path.is_file():
            raise CardNotFoundError(card_id)
        try:
            return StrategyCard.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (yaml.YAMLError, ValidationError) as exc:
            raise LibraryError(f"Card '{card_id}' is invalid: {exc}", 500) from exc

    def save(self, card: StrategyCard) -> StrategyCard:
        path = self.card_path(card.id)
        payload = yaml.safe_dump(
            card.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100
        )
        temp = path.with_suffix(".yml.tmp")
        with self._lock:
            temp.write_text(payload, encoding="utf-8")
            os.replace(temp, path)
        return card

    def _new_card_id(self, stem: str) -> str:
        base = _slug(stem)
        taken = {p.stem for p in self.cards_dir.glob("*.yml")} | set(self.reserved_names())
        candidate = f"{base}_ref"
        counter = 2
        while candidate in taken:
            candidate = f"{base}_{counter}_ref"
            counter += 1
        return candidate

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    # ------------------------------------------------------------------ upload / import

    async def upload(self, filename: str, data: bytes, *, extract: bool = True) -> UploadResult:
        upload = validate_upload(
            filename,
            data,
            allowed_types=self.config.allowed_types,
            max_bytes=self.config.max_upload_bytes,
        )
        with self._lock:
            duplicate = next((c for c in self.load_cards()[0] if c.sha256 == upload.sha256), None)
            if duplicate is not None:
                logger.info("Upload matches existing reference strategy '{}'", duplicate.id)
                return UploadResult(duplicate, duplicate=True)
            folder = self.images_dir if upload.kind.source_type == "image" else self.documents_dir
            target = folder / upload.safe_name
            counter = 2
            while target.exists():
                target = folder / f"{Path(upload.safe_name).stem}_{counter}{upload.kind.extension}"
                counter += 1
            with target.open("xb") as handle:
                handle.write(upload.data)
            stem = Path(upload.safe_name).stem
            card = StrategyCard(
                id=self._new_card_id(stem),
                name=stem.replace("_", " ").replace("-", " ").strip().title() or "Strategy",
                source_type=upload.kind.source_type,
                source_file=target.relative_to(self.root).as_posix(),
                original_filename=upload.original_name,
                media_type=upload.kind.media_type,
                sha256=upload.sha256,
                size_bytes=upload.size,
                category="pattern",
            )
            if upload.kind.source_type == "document":
                text = extract_text(upload.kind.name, upload.data).text
                self._text_cache(card.id).write_text(text, encoding="utf-8")
                if not text.strip():
                    card = card.model_copy(
                        update={"warnings": ["No text could be extracted (scanned PDF?)"]}
                    )
            self.save(card)
        logger.info(
            "Stored reference {} '{}' ({} bytes) as card '{}'",
            upload.kind.source_type,
            upload.safe_name,
            upload.size,
            card.id,
        )
        if extract:
            card = await self.run_extraction(card)
        return UploadResult(card)

    async def import_file(self, file_path: str | Path, *, extract: bool = True) -> UploadResult:
        """Import a local file (MCP tool). Only files under ``library.import_roots``."""
        name, data = await asyncio.to_thread(self._read_importable, file_path)
        return await self.upload(name, data, extract=extract)

    def _read_importable(self, file_path: str | Path) -> tuple[str, bytes]:
        # Check the allowed folders before existence, so paths outside them cannot be probed.
        try:
            resolved = Path(file_path).expanduser().resolve()
        except (OSError, RuntimeError):
            raise LibraryError(f"Invalid path: {file_path}", 400) from None
        if not any(resolved.is_relative_to(root) for root in self.import_roots):
            roots = ", ".join(str(r) for r in self.import_roots) or "(none configured)"
            raise LibraryError(
                f"'{resolved}' is outside the allowed import folders: {roots}. Copy the file "
                "there or add its folder to library.import_roots in config/app.yml.",
                403,
            )
        if not resolved.is_file():
            raise LibraryError(f"File not found: {resolved}", 404)
        if resolved.stat().st_size > self.config.max_upload_bytes:
            raise UploadError("The file exceeds the upload size limit.", 413)
        return resolved.name, resolved.read_bytes()

    # ------------------------------------------------------------------ extraction

    async def extract(self, card_id: str) -> StrategyCard:
        """Re-run extraction on user request. The card returns to draft for review."""
        card = await self.run_extraction(self.get(card_id))
        self._changed()
        return card

    async def run_extraction(self, card: StrategyCard) -> StrategyCard:
        if self.router is None:
            return self.save(self._warn(card, "No LLM configured: fill in the card manually."))
        try:
            if card.source_type == "image":
                data, media_type = self.llm_image(card)
                image = ImagePart.model_validate({"data": data, "media_type": media_type})
                request = build_image_extraction_request(image, card.original_filename)
                result = await self.router.generate_structured(
                    "vision", request, parse_card_payload
                )
            else:
                text = self.document_text(card)
                if not text.strip():
                    raise LibraryError("no text could be extracted from the document")
                request = await self._document_request(card, text)
                result = await self.router.generate_structured(
                    "extraction", request, parse_card_payload
                )
        except (LLMError, LibraryError, ValueError) as exc:
            logger.warning("Extraction failed for '{}': {}", card.id, exc)
            message = f"Extraction failed: {str(exc)[:300]}. Edit the card or re-extract."
            return self.save(self._warn(card, message))

        payload = result.value
        now = _now()
        fields = card.model_dump()
        fields.update(
            name=payload.name or card.name,
            summary=payload.summary,
            category=payload.category,
            buy_conditions=payload.buy_conditions,
            sell_conditions=payload.sell_conditions,
            sideways_conditions=payload.sideways_conditions,
            notes=payload.notes,
            status="draft",
            reviewed_by_user=False,
            extracted_at=now,
            updated_at=now,
            extracted_by=f"{result.provider} ({result.response.model})",
            warnings=[],
        )
        updated = StrategyCard.model_validate(fields).with_indicators(payload.required_indicators)
        logger.info("Extracted card '{}' via {}", updated.id, updated.extracted_by)
        return self.save(updated)

    async def _document_request(self, card: StrategyCard, text: str) -> LLMRequest:
        budget = self.config.max_document_chars
        name = card.original_filename or card.source_file
        if len(text) <= budget:
            return build_document_extraction_request(text, name, summarised=False)
        assert self.router is not None
        chunks = chunk_text(text, budget)
        if len(chunks) > MAX_CHUNKS:
            logger.warning(
                "'{}' has {} chunks; using the first {}", card.id, len(chunks), MAX_CHUNKS
            )
            chunks = chunks[:MAX_CHUNKS]
        responses = await asyncio.gather(
            *(
                self.router.generate(
                    "extraction", build_chunk_summary_request(chunk, i, len(chunks), name)
                )
                for i, chunk in enumerate(chunks, start=1)
            )
        )
        summaries = "\n\n".join(
            f"Part {i} summary:\n{r.text.strip()}" for i, r in enumerate(responses, start=1)
        )
        return build_document_extraction_request(summaries[:budget], name, summarised=True)

    @staticmethod
    def _warn(card: StrategyCard, message: str) -> StrategyCard:
        warnings = [w for w in card.warnings if not w.startswith(("Extraction failed", "No LLM"))]
        return card.model_copy(update={"warnings": [*warnings, message], "updated_at": _now()})

    # ------------------------------------------------------------------ review

    def update(self, card_id: str, changes: CardUpdate) -> StrategyCard:
        card = self.get(card_id)
        data = card.model_dump()
        edits = changes.model_dump(exclude_unset=True)
        indicators = edits.pop("required_indicators", None)
        data.update({k: v for k, v in edits.items() if v is not None})
        data.update(reviewed_by_user=True, updated_at=_now())
        updated = StrategyCard.model_validate(data)
        if indicators is not None:
            updated = updated.with_indicators(indicators)
        self.save(updated)
        if updated.status == "approved":
            self._changed()
        return updated

    def approve(self, card_id: str) -> StrategyCard:
        card = self.get(card_id)
        if not (card.summary or card.buy_conditions or card.sell_conditions):
            raise LibraryError(
                "Add a summary or at least one BUY/SELL condition before approving.", 409
            )
        try:
            self.source_path(card).stat()
        except OSError:
            raise LibraryError("The original file is missing; re-upload it.", 409) from None
        updated = card.model_copy(
            update={"status": "approved", "reviewed_by_user": True, "updated_at": _now()}
        )
        self.save(updated)
        self._changed()
        return updated

    def disable(self, card_id: str) -> StrategyCard:
        updated = self.get(card_id).model_copy(update={"status": "disabled", "updated_at": _now()})
        self.save(updated)
        self._changed()
        return updated

    def delete(self, card_id: str) -> None:
        card = self.get(card_id)
        with self._lock:
            for path in (self.card_path(card_id), self._text_cache(card_id)):
                path.unlink(missing_ok=True)
            with contextlib.suppress(LibraryError):
                self.source_path(card).unlink(missing_ok=True)
            self._image_cache.pop(card_id, None)
        logger.info("Deleted reference strategy '{}'", card_id)
        self._changed()

    # ------------------------------------------------------------------ content

    def read_source(self, card: StrategyCard) -> bytes:
        try:
            return self.source_path(card).read_bytes()
        except FileNotFoundError:
            raise LibraryError(f"Original file for '{card.id}' is missing", 404) from None

    def document_text(self, card: StrategyCard) -> str:
        """Extracted document text (cached under ``.cache``)."""
        cache = self._text_cache(card.id)
        if cache.is_file():
            return cache.read_text(encoding="utf-8")
        kind = EXTENSIONS.get(Path(card.source_file).suffix.lower().lstrip("."))
        if kind is None or KINDS[kind].source_type != "document":
            return ""
        text = extract_text(kind, self.read_source(card)).text
        cache.write_text(text, encoding="utf-8")
        return text

    def llm_image(self, card: StrategyCard) -> tuple[bytes, str]:
        """The reference image normalised for vision models (cached per file mtime)."""
        path = self.source_path(card)
        mtime = path.stat().st_mtime
        cached = self._image_cache.get(card.id)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        data, media_type = prepare_image_for_llm(path.read_bytes())
        self._image_cache[card.id] = (mtime, data, media_type)
        return data, media_type

    def preview(self, card: StrategyCard) -> dict[str, Any]:
        """What the review screen shows next to the editable card."""
        if card.source_type == "image":
            return {"kind": "image", "url": f"/api/library/{card.id}/file"}
        text = self.document_text(card)
        truncated = len(text) > PREVIEW_CHARS
        if card.media_type == "text/markdown":
            return {
                "kind": "markdown",
                "html": render_markdown(text[:PREVIEW_CHARS]),
                "truncated": truncated,
            }
        return {"kind": "text", "text": text[:PREVIEW_CHARS], "truncated": truncated}

    # ------------------------------------------------------------------ registry hook

    def strategies(self) -> tuple[list[Strategy], list[str]]:
        """Approved cards as runnable strategies (for :class:`StrategyRegistry`)."""
        cards, errors = self.load_cards()
        strategies: list[Strategy] = []
        for card in cards:
            if card.status != "approved":
                continue
            if not self.source_path(card).is_file():
                errors.append(f"card '{card.id}': original file {card.source_file} is missing")
                continue
            cls = ImageStrategy if card.source_type == "image" else DocumentStrategy
            strategies.append(cls(card, self))
        return strategies, errors


# --------------------------------------------------------------------------- strategies


class ReferenceStrategy(Strategy):
    """Common feature building for card-based strategies: the LLM applies the card."""

    min_candles = 60

    def __init__(self, card: StrategyCard, library: ReferenceLibrary) -> None:
        self.card = card
        self.library = library
        self.name = card.id
        self.display_name = card.name
        self.description = card.summary or f"Uploaded {card.source_type} strategy"
        self.category = card.category
        self.default_params = {}

    def build_features(
        self, df: pd.DataFrame, params: dict[str, Any], timeframe: str, context: StrategyContext
    ) -> FeatureBuilder:
        fb = FeatureBuilder(self, df, params, timeframe, context.history_length)
        pack = (
            context.feature_pack()
            if context.feature_pack is not None
            else compute_feature_pack(df, context.history_length)
        )
        fb.latest.update(pack["latest"])
        fb.history.update(pack["history"])
        for line in pack["summary"]:
            fb.event(line)
        fb.context["card"] = {
            "name": self.card.name,
            "summary": self.card.summary,
            "buy_conditions": self.card.buy_conditions,
            "sell_conditions": self.card.sell_conditions,
            "sideways_conditions": self.card.sideways_conditions,
            "notes": self.card.notes,
        }
        fb.context["feature_pack"] = {"swings": pack["swings"], "volume": pack["volume"]}
        if self.card.required_indicators:
            fb.context["required_indicators"] = compute_required_indicators(
                df, self.card.required_indicators, context.history_length
            )
        return fb


class DocumentStrategy(ReferenceStrategy):
    source: StrategySource = "document"

    async def analyze(
        self, df: pd.DataFrame, params: dict[str, Any], timeframe: str, context: StrategyContext
    ) -> StrategyFeatures:
        fb = await asyncio.to_thread(self.build_features, df, params, timeframe, context)
        text = await asyncio.to_thread(self.library.document_text, self.card)
        if text and len(text) <= self.library.config.max_document_chars:
            fb.context["document_text"] = text
        return fb.build()


class ImageStrategy(ReferenceStrategy):
    source: StrategySource = "image"
    requires_vision = True

    async def analyze(
        self, df: pd.DataFrame, params: dict[str, Any], timeframe: str, context: StrategyContext
    ) -> StrategyFeatures:
        fb = await asyncio.to_thread(self.build_features, df, params, timeframe, context)
        reference, media_type = await asyncio.to_thread(self.library.llm_image, self.card)
        candles = min(context.chart_candles, len(df))
        live = await render_chart_async(
            df,
            title=f"{context.instrument or 'live'} {timeframe} - last {candles} candles (UTC)",
            indicators=self.card.required_indicators,
            last_n=candles,
        )
        fb.images = [
            ImageAttachment.from_bytes("reference", reference, media_type),
            ImageAttachment.from_bytes("live_chart", live, "image/png"),
        ]
        return fb.build()
