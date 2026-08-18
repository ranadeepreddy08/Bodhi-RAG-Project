"""Language instructions for Bodhi AI Tutor."""

SUPPORTED_LANGUAGES = {
    "English": "English",
    "Telugu": "Telugu",
    "Hindi": "Hindi",
    "Tamil": "Tamil",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Bengali": "Bengali",
    "Marathi": "Marathi",
}


def get_language_instruction(language: str) -> str:

    if language == "Telugu":
        return """
Teach the student in natural, simple, conversational Telugu.

The student is a school student, so explain the concept as a
friendly Telugu-speaking teacher would explain it in class.

IMPORTANT:
- Do NOT translate an English answer word-for-word into Telugu.
- Understand the textbook concept first, then explain that concept
  naturally in Telugu.
- Use short and clear Telugu sentences.
- Avoid unnecessarily formal, literary, or difficult Telugu.
- Keep important scientific and technical terms in English when
  that makes the explanation clearer.
- When an English technical term is important, you may write the
  English term and explain its meaning naturally in Telugu.
- Simplify difficult textbook language without changing its meaning.
- Use examples or analogies only when they directly help explain
  the textbook content.
- Do NOT introduce new facts through examples or analogies.
- Stay completely grounded in the uploaded textbook.
- If the textbook does not contain enough information, do not fill
  the gap using general knowledge.
- Prefer teaching and understanding over literal translation.
"""


    if language == "English":
        return """
Teach in simple, student-friendly English.

Explain the concept as a friendly school teacher would.

- Use short, clear sentences.
- Break difficult concepts into smaller steps.
- Simplify difficult textbook wording without changing its meaning.
- Keep important technical terminology.
- Use examples or analogies only when they directly clarify the
  textbook content.
- Do not introduce outside facts.
- Stay grounded in the uploaded textbook.
"""


    return f"""
Teach the student in simple, natural {language}.

- Do not perform literal word-for-word translation.
- Explain the textbook concept naturally in the requested language.
- Use language appropriate for a school student.
- Keep important scientific and technical terms in English when
  that improves clarity.
- Simplify difficult textbook wording without changing its meaning.
- Use examples or analogies only when they directly clarify the
  uploaded textbook content.
- Do not introduce outside facts.
- Stay grounded in the uploaded textbook.
"""