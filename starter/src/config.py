"""Shared constants. One place to change paths and model names."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_ROOT / "NCERT-Class-10-Science24.pdf"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "textbook"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

LLM_MODEL = "openai/gpt-oss-20b"
