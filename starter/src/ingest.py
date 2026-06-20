"""One-time ingestion: PDF -> chunks -> embeddings -> Chroma.

Run from the project root:  python -m src.ingest

This is the file we'll build together live!
It wires together every component you've already seen:
  1. Load the PDF          (src/loaders/pdf.py)
  2. Split into chunks     (RecursiveCharacterTextSplitter)
  3. Embed the chunks      (HuggingFaceEmbeddings)
  4. Store in ChromaDB     (Chroma.from_documents)
"""
from __future__ import annotations

import shutil

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


# ---------------------------------------------------------------------------
# STEP 0 — Sanity check (already written for you — run first to verify load)
# ---------------------------------------------------------------------------

def sanity_check(pages: list[Document]) -> None:
    """Print total char count and a sample of the loaded text."""
    if not pages:
        raise ValueError(
            f"Loader returned 0 pages from {PDF_PATH}. "
            "The PDF may be empty, image-only, or corrupt."
        )
    total_chars = sum(len(p.page_content) for p in pages)
    print(f"Loaded {len(pages)} pages from {PDF_PATH.name}")
    print(f"Total characters: {total_chars:,}")
    sample = pages[0].page_content[:500].replace("\n", " ")
    print(f"\nSample (page 1, first 500 chars):\n{sample}\n")


# ---------------------------------------------------------------------------
# STEP 1 — Chunk the documents
# ---------------------------------------------------------------------------

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split each page Document into smaller overlapping chunks.

    TODO: Create a RecursiveCharacterTextSplitter with:
      - chunk_size  = CHUNK_SIZE   (imported from config, default 500)
      - chunk_overlap = CHUNK_OVERLAP (imported from config, default 50)

    Then call splitter.split_documents(docs) and return the result.

    Hint: The splitter is already imported at the top of this file.
    """
    # ✏️  YOUR CODE HERE
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# STEP 2 — Embed chunks and save to ChromaDB
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: list[Document]) -> None:
    """Embed every chunk and persist the index to disk.

    TODO:
      1. If CHROMA_DIR already exists, delete it with shutil.rmtree()
         so re-runs don't double-insert old chunks.

      2. Create a HuggingFaceEmbeddings object using EMBEDDING_MODEL.

      3. Call Chroma.from_documents() with:
           - documents  = chunks
           - embedding  = the embeddings object you just created
           - collection_name = COLLECTION_NAME
           - persist_directory = str(CHROMA_DIR)

    Hint: All names (CHROMA_DIR, EMBEDDING_MODEL, COLLECTION_NAME) are
    already imported at the top of this file.
    """
    # ✏️  YOUR CODE HERE
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


# ---------------------------------------------------------------------------
# STEP 3 — Wire it all together
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full ingestion pipeline end-to-end.

    TODO:
      1. Load the PDF:        call load_pdf(PDF_PATH)
      2. Sanity check:        call sanity_check(pages)
      3. Chunk the pages:     call chunk_documents(pages) → chunks
      4. Print chunk count.
      5. Print "Embedding and writing to Chroma..."
      6. Build the store:     call build_vectorstore(chunks)
      7. Print "Done. Index saved at {CHROMA_DIR}"
    """
    # ✏️  YOUR CODE HERE
    pages = load_pdf(PDF_PATH)
    sanity_check(pages)
    chunks = chunk_documents(pages)
    print(f"Split into {len(chunks)} chunks.")
    print("Embedding and writing to Chroma...")
    build_vectorstore(chunks)
    print(f"Done. Index saved at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
