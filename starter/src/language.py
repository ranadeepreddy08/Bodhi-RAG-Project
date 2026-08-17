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
Respond in natural, simple Telugu.

Do not perform literal word-for-word translation.
Explain the concept naturally as a Telugu-speaking teacher would.

Keep important technical terms in English when translating them
would reduce clarity.

Use simple examples and analogies suitable for a school student.
"""
    
    if language == "English":
        return """
Respond in simple, student-friendly English.
Use examples and analogies where helpful.
"""

    return f"""
Respond in simple, natural {language}.
Prefer conceptual explanation over literal word-for-word translation.
Keep important technical terms in English when appropriate.
"""