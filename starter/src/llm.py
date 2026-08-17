"""LLM layer: Groq client, prompted call, latency timing."""
from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from typing import Iterator

from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document

from src.config import LLM_MODEL
from src.prompts import SYSTEM_PROMPT, build_user_message

load_dotenv()


@dataclass(frozen=True)
class Answer:
    text: str
    latency_seconds: float


@functools.lru_cache(maxsize=1)
def _get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env "
            "and add your key."
        )

    return Groq(api_key=key)


def _build_messages(
    question: str,
    chunks: list[Document],
    language_instruction: str,
) -> list[dict]:

    user_message = build_user_message(
        question,
        chunks,
        language_instruction,
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def answer(
    question: str,
    chunks: list[Document],
    language_instruction: str,
) -> Answer:

    client = _get_client()

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=_build_messages(
            question,
            chunks,
            language_instruction,
        ),
        temperature=0,
    )

    elapsed = time.perf_counter() - start

    text = (
        response.choices[0].message.content or ""
    ).strip()

    return Answer(
        text=text,
        latency_seconds=elapsed,
    )


def stream_answer(
    question: str,
    chunks: list[Document],
    language_instruction: str,
) -> Iterator[str]:
    """Yield answer tokens as they arrive from Groq."""

    client = _get_client()

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=_build_messages(
            question,
            chunks,
            language_instruction,
        ),
        temperature=0,
        stream=True,
    )

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta.content

        if delta:
            yield delta