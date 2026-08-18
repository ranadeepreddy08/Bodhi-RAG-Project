"""Retrieval layer for the Bodhi textbook tutor."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_PATH,
    TOP_K,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model used by Bodhi."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def load_vectorstore(
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Load an existing Chroma vectorstore."""

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}."
        )

    embeddings = get_embeddings()

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def retrieve(
    query: str,
    vectorstore: Chroma,
    k: int = TOP_K,
) -> list[Document]:
    """Retrieve relevant textbook chunks."""

    if not query.strip():
        return []

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=k,
    )

    # Keep only reasonably relevant textbook chunks.
    MIN_RELEVANCE = 0.45

    relevant_docs = [
        doc
        for doc, score in results
        if score >= MIN_RELEVANCE
    ]

    return relevant_docs


def vectorstore_count(
    vectorstore: Chroma,
) -> int:
    """Return the number of chunks in the vectorstore."""

    return vectorstore._collection.count()


if __name__ == "__main__":
    vectorstore = load_vectorstore()

    print(
        "Chunks:",
        vectorstore_count(vectorstore),
    )

    docs = retrieve(
        "What is the rate limit?",
        vectorstore,
    )

    for i, doc in enumerate(docs, 1):

        page = doc.metadata.get(
            "page",
            "?",
        )

        print(
            f"--- Chunk {i} | Page {page} ---"
        )

        print(
            doc.page_content
        )

        print()