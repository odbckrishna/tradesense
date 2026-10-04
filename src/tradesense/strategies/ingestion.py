"""Ingestion of uploaded reference strategies (.md .txt .pdf .docx .png .jpg .webp).

Uploads are untrusted: the type is validated by file *signature* (not extension),
size is limited, filenames are sanitised, images are decoded with Pillow to prove
they are images, DOCX archives are checked for zip bombs, and nothing is executed.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree.ElementTree import Element

import docx
import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

FileKindName = Literal["md", "txt", "pdf", "docx", "png", "jpg", "webp"]
SourceType = Literal["document", "image"]

MAX_IMAGE_PIXELS = 40_000_000
MAX_DOCX_UNCOMPRESSED = 60 * 1024 * 1024
MAX_PDF_PAGES = 300
LLM_IMAGE_MAX_SIDE = 1568

_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class UploadError(ValueError):
    """The upload was rejected. ``status_code`` is the HTTP status to report."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class FileKind:
    name: FileKindName
    source_type: SourceType
    media_type: str
    extension: str


KINDS: dict[FileKindName, FileKind] = {
    "md": FileKind("md", "document", "text/markdown", ".md"),
    "txt": FileKind("txt", "document", "text/plain", ".txt"),
    "pdf": FileKind("pdf", "document", "application/pdf", ".pdf"),
    "docx": FileKind(
        "docx",
        "document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "png": FileKind("png", "image", "image/png", ".png"),
    "jpg": FileKind("jpg", "image", "image/jpeg", ".jpg"),
    "webp": FileKind("webp", "image", "image/webp", ".webp"),
}

EXTENSIONS: dict[str, FileKindName] = {
    "md": "md",
    "markdown": "md",
    "txt": "txt",
    "pdf": "pdf",
    "docx": "docx",
    "png": "png",
    "jpg": "jpg",
    "jpeg": "jpg",
    "webp": "webp",
}


@dataclass(frozen=True)
class ValidatedUpload:
    kind: FileKind
    original_name: str
    safe_name: str
    data: bytes
    sha256: str

    @property
    def size(self) -> int:
        return len(self.data)


# --------------------------------------------------------------------------- detection


def sniff(data: bytes) -> FileKindName | Literal["text"] | None:
    """Identify content by its signature. Text has none, so it is checked heuristically."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] == b"PK\x03\x04":
        return "docx" if _is_docx(data) else None
    if decode_text(data) is not None:
        return "text"
    return None


def _is_docx(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return False
    return "word/document.xml" in names and "[Content_Types].xml" in names


def decode_text(data: bytes) -> str | None:
    """Decode UTF-8 (or BOM-marked UTF-16) text; None if the bytes are not plain text."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None
    if "\x00" in text:
        return None
    sample = text[:20_000]
    control = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\t\n\r\f")
    if sample and control / len(sample) > 0.01:
        return None
    return text


def sanitize_filename(name: str, extension: str) -> str:
    """Strip directories and unsafe characters; ASCII, lowercase, bounded length."""
    base = Path(name.replace("\\", "/")).name
    stem = Path(base).stem
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("._-").lower()[:80] or "upload"
    if stem in _WINDOWS_RESERVED:
        stem = f"{stem}_file"
    return f"{stem}{extension}"


def validate_upload(
    filename: str,
    data: bytes,
    *,
    allowed_types: Sequence[str],
    max_bytes: int,
) -> ValidatedUpload:
    """Check size, extension and *content signature*, then deep-validate the format."""
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > max_bytes:
        raise UploadError(
            f"The file is {len(data) / 1_048_576:.1f} MB; the limit is "
            f"{max_bytes / 1_048_576:.0f} MB.",
            413,
        )
    extension = Path(filename or "").suffix.lower().lstrip(".")
    allowed = {e.lower().lstrip(".") for e in allowed_types}
    if extension not in EXTENSIONS or extension not in allowed:
        raise UploadError(
            f"File type '.{extension or '?'}' is not allowed. Allowed: "
            f"{', '.join(sorted(allowed))}.",
            415,
        )
    declared = EXTENSIONS[extension]
    detected = sniff(data)
    matches = declared in ("md", "txt") if detected == "text" else detected == declared
    if not matches:
        found = detected or "unrecognised binary data"
        raise UploadError(
            f"The file content does not match its '.{extension}' extension (detected: {found}).",
            415,
        )
    kind = KINDS[declared]
    if kind.source_type == "image":
        _check_image(data)
    elif declared == "docx":
        _check_docx(data)
    elif declared == "pdf":
        _check_pdf(data)
    return ValidatedUpload(
        kind=kind,
        original_name=Path((filename or "").replace("\\", "/")).name[:200],
        safe_name=sanitize_filename(filename or "upload", kind.extension),
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _check_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise UploadError(f"Image is too large ({width}x{height} pixels).", 413)
            image.verify()
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise UploadError(f"The image could not be decoded: {exc}", 415) from None
    except Image.DecompressionBombError:
        raise UploadError("Image is too large to process safely.", 413) from None


def _check_docx(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile:
        raise UploadError("The DOCX file is corrupt.", 415) from None
    if len(infos) > 5000 or sum(i.file_size for i in infos) > MAX_DOCX_UNCOMPRESSED:
        raise UploadError("The DOCX file expands to an unsafe size.", 413)


def _check_pdf(data: bytes) -> None:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UploadError("Encrypted PDFs are not supported.", 415)
        _ = len(reader.pages)
    except UploadError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise UploadError(f"The PDF could not be read: {exc}", 415) from None


# --------------------------------------------------------------------------- text


@dataclass(frozen=True)
class ExtractedText:
    text: str
    pages: int | None = None

    @property
    def chars(self) -> int:
        return len(self.text)


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_text(kind: FileKindName, data: bytes) -> ExtractedText:
    """Plain text of a document upload (pypdf for PDF, python-docx for DOCX)."""
    if kind == "pdf":
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = reader.pages[:MAX_PDF_PAGES]
        text = "\n\n".join((page.extract_text() or "") for page in pages)
        return ExtractedText(_normalise(text), pages=len(reader.pages))
    if kind == "docx":
        document = docx.Document(io.BytesIO(data))
        blocks = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))
        return ExtractedText(_normalise("\n\n".join(blocks)))
    if kind in ("md", "txt"):
        decoded = decode_text(data)
        if decoded is None:
            raise UploadError("The document is not valid UTF-8 text.", 415)
        return ExtractedText(_normalise(decoded))
    raise UploadError(f"'{kind}' is not a document type.", 415)


def chunk_text(text: str, max_chars: int, overlap: int = 200) -> list[str]:
    """Split on paragraph boundaries into chunks of at most ``max_chars``."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:  # very long paragraph: hard split
            head, paragraph = paragraph[:max_chars], paragraph[max_chars - overlap :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}" if tail else paragraph
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------- images


def prepare_image_for_llm(data: bytes, max_side: int = LLM_IMAGE_MAX_SIDE) -> tuple[bytes, str]:
    """Downscale large images and convert WEBP to PNG (supported by every provider).

    Returns ``(bytes, media_type)``.
    """
    with Image.open(io.BytesIO(data)) as image:
        fmt = (image.format or "").upper()
        if fmt in ("PNG", "JPEG") and max(image.size) <= max_side:
            return data, "image/png" if fmt == "PNG" else "image/jpeg"
        converted = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
        converted.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        converted.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"


# --------------------------------------------------------------------------- markdown


class _NoRawHtml(Extension):
    """Escape raw HTML instead of passing it through, and drop unsafe URLs."""

    def extendMarkdown(self, md: markdown.Markdown) -> None:
        md.preprocessors.deregister("html_block")
        md.inlinePatterns.deregister("html")
        md.treeprocessors.register(_SafeUrls(md), "safe_urls", 0)


class _SafeUrls(Treeprocessor):
    _SAFE = re.compile(r"^(https?:|mailto:|#)", re.IGNORECASE)

    def run(self, root: Element) -> None:
        for element in root.iter():
            href = element.get("href")
            if href is not None:
                if self._SAFE.match(href.strip()):
                    element.set("rel", "noopener noreferrer nofollow")
                    element.set("target", "_blank")
                else:
                    del element.attrib["href"]
            if element.tag == "img":  # never load remote images from uploaded documents
                element.attrib.pop("src", None)


def render_markdown(text: str) -> str:
    """Markdown -> HTML that is safe to insert into the review page."""
    return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", _NoRawHtml()])
