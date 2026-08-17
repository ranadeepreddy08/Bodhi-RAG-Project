"""PDF -> chunks -> embeddings -> Chroma ingestion."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_PATH,
)
from src.loaders import load_pdf


def sanity_check(
    pages: list[Document],
    pdf_path: Path | None = None,
) -> None:
    """Validate and report the extracted pages."""

    if not pages:
        source = pdf_path or PDF_PATH

        raise ValueError(
            f"Loader returned 0 pages from {source}. "
            "The PDF may be empty, image-only, or unreadable."
        )

    total_chars = sum(
        len(page.page_content)
        for page in pages
    )

    source_name = (
        pdf_path.name
        if pdf_path
        else PDF_PATH.name
    )

    print(
        f"Loaded {len(pages)} pages from {source_name}"
    )

    print(
        f"Total characters: {total_chars:,}"
    )

    sample = (
        pages[0]
        .page_content[:500]
        .replace("\n", " ")
    )

    print(
        "\nSample (page 1, first 500 chars):\n"
        f"{sample}\n"
    )


def chunk_documents(
    docs: list[Document],
) -> list[Document]:
    """Split documents into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(docs)


def build_vectorstore(
    chunks: list[Document],
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Create and persist a Chroma vectorstore."""

    if persist_directory.exists():
        shutil.rmtree(persist_directory)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )

    return vectorstore


def build_index(
    pdf_path: Path,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Build a complete vectorstore from a PDF."""

    print(f"Loading PDF: {pdf_path}")

    pages = load_pdf(pdf_path)

    sanity_check(
        pages,
        pdf_path,
    )

    chunks = chunk_documents(pages)

    print(
        f"Split into {len(chunks)} chunks."
    )

    print(
        "Embedding and writing to Chroma..."
    )

    vectorstore = build_vectorstore(
        chunks,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(
        f"Done. Index saved at {persist_directory}"
    )

    return vectorstore


def main() -> None:
    """Run the original command-line ingestion."""

    build_index(
        PDF_PATH,
        CHROMA_DIR,
        COLLECTION_NAME,
    )


if __name__ == "__main__":
    main()