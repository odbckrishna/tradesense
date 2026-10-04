from __future__ import annotations

import io
import zipfile

import pytest
from PIL import Image

from tests.helpers import make_docx, make_image, make_pdf
from tradesense.strategies.ingestion import (
    UploadError,
    chunk_text,
    extract_text,
    prepare_image_for_llm,
    render_markdown,
    sanitize_filename,
    sniff,
    validate_upload,
)

ALLOWED = ["md", "txt", "pdf", "docx", "png", "jpg", "jpeg", "webp"]
MB = 1024 * 1024


def check(name: str, data: bytes, max_bytes: int = 10 * MB):  # type: ignore[no-untyped-def]
    return validate_upload(name, data, allowed_types=ALLOWED, max_bytes=max_bytes)


# --------------------------------------------------------------------------- happy paths


def test_markdown_ingestion() -> None:
    upload = check("RSI Rules.md", b"# RSI rules\n\nBuy below **30**.\n")
    assert upload.kind.name == "md"
    assert upload.kind.source_type == "document"
    assert upload.safe_name == "rsi_rules.md"
    assert extract_text("md", upload.data).text.startswith("# RSI rules")


def test_pdf_ingestion() -> None:
    data = make_pdf(["Double bottom strategy", "Buy on a close above the neckline."])
    upload = check("strategy.pdf", data)
    assert upload.kind.media_type == "application/pdf"
    extracted = extract_text("pdf", data)
    assert "close above the neckline" in extracted.text
    assert extracted.pages == 1


def test_docx_ingestion_includes_tables() -> None:
    data = make_docx(
        ["Breakout strategy", "Enter when price closes above resistance."],
        table=[["Indicator", "Setting"], ["EMA", "20"]],
    )
    upload = check("breakout.docx", data)
    assert upload.kind.name == "docx"
    text = extract_text("docx", data).text
    assert "closes above resistance" in text
    assert "EMA | 20" in text


@pytest.mark.parametrize(
    ("fmt", "name", "kind"),
    [
        ("PNG", "chart.png", "png"),
        ("JPEG", "chart.jpeg", "jpg"),
        ("WEBP", "chart.webp", "webp"),
    ],
)
def test_image_ingestion(fmt: str, name: str, kind: str) -> None:
    upload = check(name, make_image(fmt))
    assert upload.kind.name == kind
    assert upload.kind.source_type == "image"


def test_text_with_utf16_bom() -> None:
    data = "Buy when RSI < 30".encode("utf-16")
    assert sniff(data) == "text"
    assert check("rules.txt", data).kind.name == "txt"


# --------------------------------------------------------------------------- rejections


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("chart.pdf", make_image("PNG"), "detected: png"),
        ("notes.png", b"# just markdown", "does not match"),
        ("script.md", b"MZ\x90\x00\x03\x00\x00\x00\x04\x00", "does not match"),
        ("payload.exe", b"MZ\x90\x00", "not allowed"),
        ("noext", b"hello", "not allowed"),
        ("empty.md", b"", "empty"),
    ],
)
def test_signature_and_type_checks(name: str, data: bytes, message: str) -> None:
    with pytest.raises(UploadError, match=message):
        check(name, data)


def test_zip_that_is_not_docx_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("evil.sh", "rm -rf /")
    with pytest.raises(UploadError, match="does not match"):
        check("report.docx", buffer.getvalue())


def test_size_limit() -> None:
    with pytest.raises(UploadError, match="limit") as info:
        check("big.md", b"a" * (2 * MB), max_bytes=MB)
    assert info.value.status_code == 413


def test_truncated_image_is_rejected() -> None:
    data = make_image("PNG")[:60]
    with pytest.raises(UploadError, match="could not be decoded"):
        check("broken.png", data)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.md", "passwd.md"),
        ("C:\\Windows\\System32\\evil.md", "evil.md"),
        ("Trend Following (v2).md", "trend_following_v2.md"),
        ("éàü stratégie.md", "eau_strategie.md"),
        ("CON.md", "con_file.md"),
        ("....md", "upload.md"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw, ".md") == expected


# --------------------------------------------------------------------------- utilities


def test_chunking_respects_limits() -> None:
    paragraphs = [f"Rule {i}: " + "x" * 300 for i in range(40)]
    chunks = chunk_text("\n\n".join(paragraphs), max_chars=2000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 + 100 for c in chunks)
    assert "Rule 0:" in chunks[0]
    assert "Rule 39:" in chunks[-1]
    assert chunk_text("short", 2000) == ["short"]


def test_prepare_image_for_llm() -> None:
    small = make_image("PNG")
    assert prepare_image_for_llm(small) == (small, "image/png")
    data, media_type = prepare_image_for_llm(make_image("WEBP"))
    assert media_type == "image/png"
    assert data.startswith(b"\x89PNG")
    big, _ = prepare_image_for_llm(make_image("PNG", (3000, 1000)))
    with Image.open(io.BytesIO(big)) as image:
        assert max(image.size) == 1568


def test_markdown_rendering_is_sanitised() -> None:
    html = render_markdown(
        "# Title\n<script>alert(1)</script>\n\n[x](javascript:alert(1)) [ok](https://a.b)\n"
        "![t](http://tracker/p.png)\n\n<img src=x onerror=alert(1)>"
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "javascript:" not in html
    assert 'href="https://a.b"' in html
    assert "tracker" not in html
    assert "<img src=x" not in html
