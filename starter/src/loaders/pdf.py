"""Robust PDF loader for Bodhi.

Extraction order:
1. PyMuPDF text extraction
2. pdfplumber text extraction
3. OCR fallback using Tesseract

Tables are also extracted with pdfplumber when available.
"""

from __future__ import annotations

from pathlib import Path
import re

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Windows OCR configuration
# ---------------------------------------------------------------------------

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POPPLER_PATH = (
    r"C:\Release-26.02.0-0"
    r"\poppler-26.02.0"
    r"\Library"
    r"\bin"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------

def _serialize_table(
    table: list[list[str | None]],
) -> str:
    """Convert a pdfplumber table into searchable text."""

    if not table or len(table) < 2:
        return ""

    headers = [
        (header or "").strip()
        for header in table[0]
    ]

    lines: list[str] = []

    for row in table[1:]:
        cells = []

        for header, raw in zip(headers, row):
            value = (raw or "").strip()

            if not value:
                continue

            if header:
                cells.append(
                    f"{header}: {value}"
                )
            else:
                cells.append(value)

        if cells:
            lines.append(" | ".join(cells))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text quality detection
# ---------------------------------------------------------------------------

def _looks_like_real_text(text: str) -> bool:
    """Return True when extracted text appears usable."""

    if not text:
        return False

    text = text.strip()

    if len(text) < 30:
        return False

    # Count normal letters/numbers.
    normal_chars = sum(
        1
        for char in text
        if char.isalnum()
    )

    normal_ratio = normal_chars / max(len(text), 1)

    # Most textbook prose should contain a reasonable amount
    # of letters/numbers.
    if normal_ratio < 0.25:
        return False

    # Detect common font-encoding garbage.
    suspicious_symbols = re.findall(
        r"[❈✁✂✃✄☎✆✇✈✉✐✒✓✔✕✖✗✘✙✚✛✜✢✣✤✥✦✧♦]",
        text,
    )

    if len(suspicious_symbols) >= 5:
        return False

    return True


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _ocr_page(
    pdf_path: Path,
    page_number: int,
) -> str:
    """OCR one PDF page using Tesseract."""

    images = convert_from_path(
        str(pdf_path),
        dpi=250,
        first_page=page_number,
        last_page=page_number,
        poppler_path=POPPLER_PATH,
    )

    if not images:
        return ""

    return pytesseract.image_to_string(
        images[0],
        lang="eng",
    ).strip()


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_pdf(path: Path) -> list[Document]:
    """Load a PDF into one LangChain Document per page."""

    if not path.exists():
        raise FileNotFoundError(
            f"Source PDF not found: {path}"
        )

    source_name = (
        f"{path.parent.name}/{path.name}"
        if path.parent.name
        else path.name
    )

    documents: list[Document] = []

    # =======================================================================
    # METHOD 1 — PyMuPDF
    # =======================================================================

    print("Trying PyMuPDF extraction...")

    try:
        pymupdf_doc = fitz.open(str(path))

        for page_index, page in enumerate(pymupdf_doc):

            text = page.get_text(
                "text"
            ).strip()

            if _looks_like_real_text(text):

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source_name,
                            "page": page_index,
                            "extraction": "pymupdf",
                        },
                    )
                )

            else:
                print(
                    f"Page {page_index + 1}: "
                    "PyMuPDF text unavailable/corrupted."
                )

        pymupdf_doc.close()

    except Exception as exc:
        print(
            f"PyMuPDF extraction failed: {exc}"
        )

    # If PyMuPDF successfully extracted every page, return it.
    try:
        page_count = len(
            fitz.open(str(path))
        )
    except Exception:
        page_count = 0

    if (
        page_count > 0
        and len(documents) == page_count
    ):
        print(
            f"PyMuPDF successfully extracted "
            f"{len(documents)} pages."
        )

        return documents

    # =======================================================================
    # METHOD 2 — pdfplumber
    # =======================================================================

    print(
        "Trying pdfplumber extraction..."
    )

    plumber_documents: list[Document] = []

    try:
        with pdfplumber.open(str(path)) as pdf:

            for page_index, page in enumerate(
                pdf.pages
            ):

                prose = (
                    page.extract_text()
                    or ""
                ).strip()

                table_blocks: list[str] = []

                for table in (
                    page.extract_tables()
                    or []
                ):
                    serialized = (
                        _serialize_table(table)
                    )

                    if serialized:
                        table_blocks.append(
                            serialized
                        )

                tables_text = ""

                if table_blocks:
                    tables_text = (
                        "\n\n[TABLES]\n"
                        + "\n\n".join(
                            table_blocks
                        )
                    )

                page_text = (
                    prose + tables_text
                ).strip()

                if _looks_like_real_text(
                    page_text
                ):
                    plumber_documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": source_name,
                                "page": page_index,
                                "extraction": (
                                    "pdfplumber"
                                ),
                            },
                        )
                    )

    except Exception as exc:
        print(
            f"pdfplumber extraction failed: {exc}"
        )

    if plumber_documents:
        print(
            f"pdfplumber extracted "
            f"{len(plumber_documents)} usable pages."
        )

        return plumber_documents

    # =======================================================================
    # METHOD 3 — OCR
    # =======================================================================

    print(
        "Normal extraction failed. "
        "Trying OCR..."
    )

    ocr_documents: list[Document] = []

    for page_index in range(page_count):

        page_number = page_index + 1

        print(
            f"OCR page {page_number}/{page_count}..."
        )

        try:
            text = _ocr_page(
                path,
                page_number,
            )

        except Exception as exc:
            print(
                f"OCR failed on page "
                f"{page_number}: {exc}"
            )
            text = ""

        if _looks_like_real_text(text):

            ocr_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_index,
                        "extraction": "tesseract",
                    },
                )
            )

    if ocr_documents:
        print(
            f"OCR extracted "
            f"{len(ocr_documents)} usable pages."
        )

    return ocr_documents