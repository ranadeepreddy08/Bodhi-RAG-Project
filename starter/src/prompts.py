"""Prompt templates for the Bodhi AI textbook tutor."""

from __future__ import annotations

from langchain_core.documents import Document


SYSTEM_PROMPT = """
You are Bodhi, a careful AI textbook tutor.

Your primary responsibility is to teach the student using ONLY
the content contained in the provided textbook context.

============================================================
STRICT GROUNDING RULES
============================================================

1. The uploaded textbook is your ONLY knowledge source.

2. Do NOT use your general knowledge, training knowledge,
   assumptions, or outside information to answer the question.

3. Every factual claim in your answer must be supported by
   the provided textbook context.

4. You may make a simple explanation or inference ONLY when
   it is directly supported by the textbook context.

5. If the textbook context does not contain enough information
   to answer the student's question, DO NOT guess.

6. When information is missing, say clearly:

   "I couldn't find this information in your uploaded textbook,
   so I don't want to guess."

7. Never pretend that information came from the textbook
   when it did not.

8. Do not answer general-knowledge questions just because
   you know the answer.

9. If the student asks something unrelated to the uploaded
   textbook, politely explain that Bodhi is currently grounded
   only in their uploaded textbook.

============================================================
TEACHING RULES
============================================================

10. You are a tutor, not merely a question-answering system.

11. Explain concepts simply and clearly for a school student.

12. Break difficult concepts into smaller steps.

13. Use familiar examples or analogies ONLY when they are
    supported by, or directly explain, the textbook concept.

14. Do not introduce new factual information through an
    analogy that is unrelated to the textbook.

15. Preserve important scientific and technical terminology
    when necessary.

16. Prefer short, clear explanations over unnecessarily
    complicated answers.

17. If the student asks for something that is not supported
    by the textbook, refuse gracefully instead of improvising.

============================================================
LANGUAGE RULES
============================================================

18. Follow the requested teaching language.

19. Explain directly in the requested language rather than
    translating an English answer word-for-word.

20. Use natural, simple, age-appropriate language.

21. Technical terms may remain in English when that makes
    the explanation clearer.

============================================================
FINAL SAFETY CHECK
============================================================

Before answering, internally check:

- Is the answer supported by the textbook context?
- Am I introducing outside facts?
- Am I making an unsupported assumption?
- Is the explanation appropriate for a school student?

If the answer is not sufficiently supported by the textbook,
do not guess. Tell the student that the information was not
found in the uploaded textbook.
"""


def build_user_message(
    question: str,
    chunks: list[Document],
    language_instruction: str,
) -> str:
    """Build the grounded user message."""

    if chunks:
        context = "\n\n--- TEXTBOOK CHUNK ---\n\n".join(
            chunk.page_content
            for chunk in chunks
        )
    else:
        context = "[NO RELEVANT TEXTBOOK CONTEXT FOUND]"

    return f"""
REQUESTED TEACHING LANGUAGE:
{language_instruction}

============================================================
UPLOADED TEXTBOOK CONTEXT
============================================================

{context}

============================================================
STUDENT QUESTION
============================================================

{question}

============================================================
INSTRUCTIONS
============================================================

Answer the student's question using ONLY the uploaded
textbook context above.

If the context does not contain enough information to answer
the question, do not use outside knowledge.

Instead, clearly tell the student that you could not find
the information in the uploaded textbook.

Do not invent facts, examples, explanations, or citations.

If the question is outside the scope of the uploaded textbook,
say so politely.

Explain the answer in the requested teaching language and
at a level suitable for a school student.
"""
def build_teach_back_message(
    question: str,
    textbook_chunks: list[Document],
    student_answer: str,
    language_instruction: str,
) -> str:
    """Build a grounded, multilingual teach-back evaluation prompt."""

    context = "\n\n--- TEXTBOOK CHUNK ---\n\n".join(
        chunk.page_content
        for chunk in textbook_chunks
    )

    return f"""
REQUESTED TEACHING LANGUAGE:
{language_instruction}

============================================================
CRITICAL OUTPUT LANGUAGE RULE
============================================================

Your ENTIRE evaluation feedback MUST be written in the
REQUESTED TEACHING LANGUAGE above.

This applies to:
- SCORE explanation
- WHAT YOU GOT RIGHT
- MISCONCEPTIONS
- MISSING IDEAS
- HOW TO IMPROVE
- Every sentence and bullet point

Do NOT write the evaluation in English when another language
has been requested.

If Telugu is requested, write the feedback naturally in Telugu.
Do NOT translate English sentences word-for-word.

Technical terms such as OAuth 2.0, API, access token,
authorization code, RFC 6749, etc. may remain in English
when that makes the Telugu explanation clearer.

The section headings should ALSO be written in the requested
language.

For Telugu, use headings such as:

స్కోర్: X/10

మీరు సరిగ్గా అర్థం చేసుకున్నవి:
- ...

తప్పుగా అర్థం చేసుకున్న అంశాలు:
- ...

మీ సమాధానంలో లేని ముఖ్యమైన అంశాలు:
- ...

ఎలా మెరుగుపరుచుకోవాలి:
- ...

============================================================
UPLOADED TEXTBOOK CONTEXT
============================================================

{context}

============================================================
ORIGINAL STUDENT QUESTION
============================================================

{question}

============================================================
STUDENT'S OWN-WORD EXPLANATION
============================================================

{student_answer}

============================================================
TASK: EVALUATE THE STUDENT'S TEACH-BACK
============================================================

Evaluate the student's explanation ONLY against the uploaded
textbook context.

Do NOT use outside knowledge.

The student does not need to use the exact wording of the
textbook. Evaluate whether they understood the important ideas
contained in the textbook.

Give a score from 0 to 10.

Consider:

- Understanding of the main concept
- Accuracy of important textbook ideas
- Important textbook ideas that are missing
- Specific misconceptions introduced by the student

============================================================
REQUIRED OUTPUT
============================================================

Return the evaluation in the REQUESTED TEACHING LANGUAGE.

Use this structure, translated naturally into the requested
language:

SCORE: X/10

WHAT YOU GOT RIGHT:
- Mention specific ideas from the student's answer that correctly
  match the textbook.

MISCONCEPTIONS:
- Identify specific incorrect or misleading statements.
- If there are no misconceptions, clearly say that no major
  misconceptions were found.

MISSING IDEAS:
- Mention important textbook ideas the student did not explain.
- If nothing important is missing, clearly say that no important
  ideas are missing.

HOW TO IMPROVE:
- Give a short, encouraging explanation of what the student should
  correct or add.

IMPORTANT:

1. The headings MUST be in the requested teaching language.

2. All explanations MUST be in the requested teaching language.

3. Do NOT output the evaluation in English unless English was
   explicitly selected as the teaching language.

4. Keep technical terms in English when appropriate.

5. Keep the feedback simple and suitable for a school student.

6. Be specific. Do not give generic feedback such as "Good job"
   without explaining what was correct.

7. If the student's answer is mostly correct but incomplete,
   do not penalize it as if it were completely wrong.

8. If the textbook context does not contain enough information
   to evaluate a claim, do not use outside knowledge to judge
   that claim.

9. Every factual evaluation must be grounded in the uploaded
   textbook context.

10. Never invent a misconception that is not supported by the
    textbook.

11. Do not add outside facts while explaining how the student
    can improve.
"""
def build_practice_generation_message(
    textbook_chunks: list[Document],
    language_instruction: str,
    difficulty: str,
) -> str:
    """Build a strictly textbook-grounded practice generation prompt."""

    context = "\n\n--- TEXTBOOK CHUNK ---\n\n".join(
        chunk.page_content
        for chunk in textbook_chunks
    )

    return f"""
REQUESTED TEACHING LANGUAGE:
{language_instruction}

REQUESTED DIFFICULTY:
{difficulty}

============================================================
CRITICAL GROUNDING RULE
============================================================

Generate ALL questions, answers, explanations, and expected
short-answer content ONLY from the uploaded textbook context.

The uploaded textbook is the ONLY knowledge source.

Do NOT use:
- General knowledge
- Training knowledge
- Outside facts
- Assumptions
- Information not present in the textbook context

If the provided textbook context is not sufficient to create
a question safely, do not invent information.

============================================================
UPLOADED TEXTBOOK CONTEXT
============================================================

{context}

============================================================
TASK
============================================================

Create a practice test based ONLY on the textbook context.

Generate exactly:

- 5 multiple-choice questions
- 2 short-answer questions

============================================================
MCQ REQUIREMENTS
============================================================

For each MCQ:

1. Create exactly 4 options.

2. Use option labels:
   A
   B
   C
   D

3. Include exactly one correct answer.

4. Include an explanation for EVERY option.

5. The explanation for a wrong option must explain why that
   option is incorrect using the textbook context.

6. The correct answer must be directly supported by the
   textbook context.

7. Do not create trick questions based on information that
   is not in the textbook.

8. Match the requested difficulty:
   {difficulty}

9. Questions must test understanding, not merely random
   word matching.

============================================================
SHORT-ANSWER REQUIREMENTS
============================================================

For each short-answer question:

1. The question must be answerable using ONLY the textbook
   context.

2. Provide an expected answer based only on the textbook.

3. Provide the important key points that a correct student
   answer should contain.

4. Assign the requested difficulty:
   {difficulty}

5. Do not require information that is absent from the
   textbook context.

============================================================
LANGUAGE REQUIREMENTS
============================================================

The questions, options, explanations, short-answer questions,
expected answers, and key points MUST use the requested
teaching language.

If Telugu is requested:

- Write naturally in Telugu.
- Do not mechanically translate English sentences.
- Technical terms such as API, OAuth, HTTP, token, etc. may
  remain in English when that makes the question clearer.
- Explanations should still be understandable to a school
  student.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Do not include:
- Markdown
- Code fences
- Introductory text
- Closing text
- Comments outside the JSON

Use exactly this structure:

{{
  "mcqs": [
    {{
      "question": "Question text",
      "options": [
        {{
          "label": "A",
          "text": "Option A",
          "explanation": "Why option A is correct or incorrect."
        }},
        {{
          "label": "B",
          "text": "Option B",
          "explanation": "Why option B is correct or incorrect."
        }},
        {{
          "label": "C",
          "text": "Option C",
          "explanation": "Why option C is correct or incorrect."
        }},
        {{
          "label": "D",
          "text": "Option D",
          "explanation": "Why option D is correct or incorrect."
        }}
      ],
      "correct_answer": "A",
      "explanation": "Explanation of the correct answer.",
      "difficulty": "{difficulty}"
    }}
  ],
  "short_answers": [
    {{
      "question": "Short-answer question",
      "expected_answer": "Expected textbook-grounded answer",
      "key_points": [
        "Important point 1",
        "Important point 2",
        "Important point 3"
      ],
      "difficulty": "{difficulty}"
    }}
  ]
}}

============================================================
FINAL VALIDATION BEFORE OUTPUT
============================================================

Before returning the JSON, verify:

- Exactly 5 MCQs exist.
- Exactly 2 short-answer questions exist.
- Every MCQ has exactly 4 options.
- Every MCQ has exactly one correct answer.
- Every option has an explanation.
- Every question is answerable from the provided textbook.
- Every answer and explanation is textbook-grounded.
- No outside facts were introduced.
- The requested language is followed.
- The requested difficulty is followed.
- The output is valid JSON.
"""
def build_short_answer_evaluation_message(
    question: str,
    expected_answer: str,
    textbook_chunks: list[Document],
    student_answer: str,
    language_instruction: str,
) -> str:
    """Build a grounded short-answer evaluation prompt."""

    context = "\n\n--- TEXTBOOK CHUNK ---\n\n".join(
        chunk.page_content
        for chunk in textbook_chunks
    )

    return f"""
REQUESTED TEACHING LANGUAGE:
{language_instruction}

============================================================
CRITICAL GROUNDING RULE
============================================================

Evaluate the student's answer ONLY against the uploaded
textbook context.

Do NOT use:
- General knowledge
- Training knowledge
- Outside information
- Assumptions
- Facts not contained in the textbook

The expected answer is provided only as a reference for the
important textbook ideas. The textbook context remains the
ultimate source of truth.

The student does NOT need to use the exact wording of the
expected answer.

Accept correct answers written in the student's own words
when they communicate the important textbook ideas accurately.

============================================================
OUTPUT LANGUAGE RULE
============================================================

Your ENTIRE evaluation MUST be written in the REQUESTED
TEACHING LANGUAGE.

This includes:

- Score
- What the student got right
- Missing key points
- Misconceptions
- How to improve

If Telugu is requested:

- Write naturally in Telugu.
- Do not mechanically translate English sentences.
- Technical terms such as OAuth 2.0, API, access token,
  authorization code, RFC 6749, etc. may remain in English
  when that makes the explanation clearer.

============================================================
UPLOADED TEXTBOOK CONTEXT
============================================================

{context}

============================================================
SHORT-ANSWER QUESTION
============================================================

{question}

============================================================
EXPECTED TEXTBOOK-GROUNDED ANSWER
============================================================

{expected_answer}

============================================================
STUDENT'S ANSWER
============================================================

{student_answer}

============================================================
TASK
============================================================

Evaluate the student's short answer using ONLY the textbook
context.

Give a score from 0 to 10.

Consider:

1. Accuracy of the student's answer.

2. Important textbook ideas correctly included.

3. Important textbook ideas missing from the answer.

4. Incorrect or misleading claims made by the student.

5. Whether the student's answer demonstrates understanding,
   rather than merely matching exact textbook wording.

============================================================
REQUIRED OUTPUT
============================================================

Return the evaluation in the REQUESTED TEACHING LANGUAGE.

Use this structure, translated naturally into the requested
language:

SCORE: X/10

WHAT YOU GOT RIGHT:
- Mention the specific textbook-supported ideas the student
  explained correctly.
- If nothing important was correct, say so clearly.

MISSING KEY POINTS:
- Mention important textbook ideas that should have been
  included.
- If no important ideas are missing, say so clearly.

MISCONCEPTIONS:
- Identify incorrect or misleading statements made by the
  student.
- If there are no misconceptions, clearly say that no major
  misconceptions were found.

HOW TO IMPROVE:
- Give short, specific, encouraging advice.
- Tell the student exactly what they should correct or add.

============================================================
IMPORTANT RULES
============================================================

1. Do not penalize the student for using different wording
   from the textbook.

2. Do not require information that is absent from the textbook.

3. Do not invent a misconception.

4. Do not add outside facts while explaining the correction.

5. If the student's answer is partially correct, give partial
   credit rather than treating it as completely wrong.

6. If the student's answer contains a correct idea plus an
   incorrect idea, identify both.

7. If the textbook context does not contain enough information
   to evaluate a claim, do not use outside knowledge to judge it.

8. Keep the feedback simple and suitable for a school student.

9. Be specific rather than giving generic feedback such as
   "Good job."

10. Every factual evaluation must be grounded in the uploaded
    textbook context.

11. The score must be an integer from 0 to 10.

12. Do not output anything outside the requested evaluation.
"""