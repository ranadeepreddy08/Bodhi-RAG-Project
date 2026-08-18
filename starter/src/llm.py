"""LLM layer: Groq client, prompted call, latency timing."""
from __future__ import annotations

import functools
import json
import os
import time
from dataclasses import dataclass
from typing import Iterator

from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document

from src.config import LLM_MODEL
from src.prompts import (
    SYSTEM_PROMPT,
    build_user_message,
    build_practice_generation_message,
)

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

def evaluate_teach_back(
    question: str,
    chunks: list[Document],
    student_answer: str,
    language_instruction: str,
) -> Answer:
    """Evaluate a student's teach-back answer using textbook context."""

    from src.prompts import build_teach_back_message

    client = _get_client()

    start = time.perf_counter()

    user_message = build_teach_back_message(
        question,
        chunks,
        student_answer,
        language_instruction,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
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
@dataclass(frozen=True)
class PracticeGeneration:
    """Generated adaptive practice content."""

    data: dict
    latency_seconds: float


def generate_practice(
    chunks: list[Document],
    language_instruction: str,
    difficulty: str,
) -> PracticeGeneration:
    """Generate textbook-grounded practice questions using Groq."""

    client = _get_client()

    start = time.perf_counter()

    user_message = f"""
Create exactly 5 multiple-choice questions from the textbook
context below.

Create exactly 2 short-answer questions.

All 7 questions must have this difficulty:

{difficulty}

Return ONLY valid JSON using this structure:

{{
  "mcqs": [
    {{
      "question": "question",
      "options": [
        {{
          "label": "A",
          "text": "option text",
          "explanation": "Explain why this option is correct or incorrect using only the textbook."
        }},
        {{
          "label": "B",
          "text": "option text",
          "explanation": "Explain why this option is correct or incorrect using only the textbook."
        }},
        {{
          "label": "C",
          "text": "option text",
          "explanation": "Explain why this option is correct or incorrect using only the textbook."
        }},
        {{
          "label": "D",
          "text": "option text",
          "explanation": "Explain why this option is correct or incorrect using only the textbook."
        }}
      ],
      "correct_answer": "A",
      "difficulty": "{difficulty}"
    }}
  ],
  "short_answers": [
    {{
      "question": "question",
      "expected_answer": "answer",
      "difficulty": "{difficulty}"
    }}
  ]
}}

============================================================
GROUNDING RULE
============================================================

Use ONLY the textbook context below.

Do NOT use:
- General knowledge
- Outside knowledge
- Training knowledge
- Assumptions
- Facts not contained in the textbook

Every question, option, correct answer, and explanation must
be supported by the textbook context.

============================================================
MCQ REQUIREMENTS
============================================================

For every MCQ:

- There must be exactly 4 options.
- Exactly one option must be correct.
- Every option must have an explanation.
- Explain why the correct option is correct.
- Explain why every wrong option is incorrect.
- Do not invent information to explain a wrong option.
- Difficulty must be exactly "{difficulty}".

============================================================
SHORT-ANSWER REQUIREMENTS
============================================================

For every short-answer question:

- The question must be answerable from the textbook.
- The expected answer must contain only textbook-supported
  information.
- Difficulty must be exactly "{difficulty}".

============================================================
LANGUAGE
============================================================

Generate the questions, options, explanations, and expected
answers in the requested teaching language.

Language: {language_instruction}

============================================================
TEXTBOOK CONTEXT
============================================================

{chr(10).join(chunk.page_content for chunk in chunks[:3])}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0,
        max_completion_tokens=12000,
        response_format={"type": "json_object"},
    )

    elapsed = time.perf_counter() - start

    text = (
        response.choices[0].message.content or ""
    ).strip()
    print("DEBUG finish_reason:", response.choices[0].finish_reason)
    print("DEBUG raw length:", len(text))
    print("DEBUG raw tail:", text[-300:])
    if not text:
        raise RuntimeError(
            "Groq returned an empty practice response."
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Groq returned invalid JSON for practice generation."
        ) from exc


    if not isinstance(data, dict):
        raise RuntimeError(
            "Practice generation response must be a JSON object."
        )

    if "mcqs" not in data or "short_answers" not in data:
        raise RuntimeError(
            "Practice generation response is missing required sections."
        )

    if len(data["mcqs"]) != 5:
        raise RuntimeError(
            f"Expected 5 MCQs, but received {len(data['mcqs'])}."
        )

    if len(data["short_answers"]) < 1:
        raise RuntimeError(
            "No short-answer questions were generated."
        )

    allowed_difficulties = {"Easy", "Medium", "Hard"}

    for index, mcq in enumerate(data["mcqs"], start=1):

        options = mcq.get("options", [])

        if len(options) != 4:
            raise RuntimeError(
                f"MCQ {index} must contain exactly 4 options."
            )

        for option_index, option in enumerate(options, start=1):

            if not isinstance(option, dict):
                raise RuntimeError(
                    f"MCQ {index}, option {option_index} "
                    "must be an object."
                )

            if not option.get("label"):
                raise RuntimeError(
                    f"MCQ {index}, option {option_index} "
                    "is missing its label."
                )

            if not option.get("text"):
                raise RuntimeError(
                    f"MCQ {index}, option {option_index} "
                    "is missing its text."
                )

            if not option.get("explanation"):
                raise RuntimeError(
                    f"MCQ {index}, option {option_index} "
                    "is missing its explanation."
                )

        correct_answer = mcq.get("correct_answer")

        if correct_answer not in {"A", "B", "C", "D"}:
            raise RuntimeError(
                f"MCQ {index} has an invalid correct answer."
            )

        mcq_difficulty = mcq.get("difficulty")

        if mcq_difficulty not in allowed_difficulties:
            raise RuntimeError(
                f"MCQ {index} has an invalid difficulty: "
                f"{mcq_difficulty}"
            )

        if mcq_difficulty != difficulty:
            raise RuntimeError(
                f"MCQ {index} difficulty is "
                f"{mcq_difficulty}, expected {difficulty}."
            )

    for index, short_answer in enumerate(
        data["short_answers"],
        start=1,
    ):

        if not short_answer.get("question"):
            raise RuntimeError(
                f"Short-answer question {index} is missing its question."
            )

        if not short_answer.get("expected_answer"):
            raise RuntimeError(
                f"Short-answer question {index} "
                "is missing its expected answer."
            )

        short_answer_difficulty = short_answer.get("difficulty")

        if short_answer_difficulty not in allowed_difficulties:
            raise RuntimeError(
                f"Short-answer question {index} has an invalid "
                f"difficulty: {short_answer_difficulty}"
            )

        if short_answer_difficulty != difficulty:
            raise RuntimeError(
                f"Short-answer question {index} difficulty is "
                f"{short_answer_difficulty}, expected {difficulty}."
            )

    return PracticeGeneration(
        data=data,
        latency_seconds=elapsed,
    )
def evaluate_short_answer(
    question: str,
    expected_answer: str,
    chunks: list[Document],
    student_answer: str,
    language_instruction: str,
) -> Answer:
    """Evaluate a student's short answer using textbook grounding."""

    from src.prompts import build_short_answer_evaluation_message

    client = _get_client()

    start = time.perf_counter()

    user_message = build_short_answer_evaluation_message(
        question,
        expected_answer,
        chunks,
        student_answer,
        language_instruction,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
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