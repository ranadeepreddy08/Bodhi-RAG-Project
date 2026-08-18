"""Adaptive practice question models and difficulty logic."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MCQOption:
    """One MCQ option with its explanation."""

    label: str
    text: str
    explanation: str


@dataclass
class MCQ:
    """A multiple-choice practice question."""

    question: str
    options: List[MCQOption]
    correct_answer: str
    explanation: str
    difficulty: str
    source: Optional[str] = None


@dataclass
class ShortAnswerQuestion:
    """A textbook-grounded short-answer question."""

    question: str
    expected_answer: str
    key_points: List[str]
    difficulty: str
    source: Optional[str] = None


@dataclass
class ShortAnswerEvaluation:
    """Evaluation result for a short-answer response."""

    score: int
    feedback: str
    missing_points: List[str]
    misconceptions: List[str]


def calculate_next_difficulty(
    correct_answers: int,
    total_questions: int = 5,
) -> str:
    """
    Determine the next difficulty level after an MCQ round.

    Rules:
        4-5 correct -> Hard
        2-3 correct -> Medium
        0-1 correct -> Easy

    The default practice round contains 5 MCQs.
    """

    if total_questions <= 0:
        return "Medium"

    if correct_answers < 0:
        correct_answers = 0

    if correct_answers > total_questions:
        correct_answers = total_questions

    if correct_answers >= 4:
        return "Hard"

    if correct_answers >= 2:
        return "Medium"

    return "Easy"


def validate_difficulty(difficulty: str) -> str:
    """Normalize and validate a difficulty value."""

    normalized = difficulty.strip().capitalize()

    if normalized not in {"Easy", "Medium", "Hard"}:
        return "Medium"

    return normalized