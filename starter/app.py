"""Bodhi AI Tutor - Streamlit application."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
from groq import APIError, APIConnectionError, RateLimitError

from src import ingest, llm, rag
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
)
from src.language import (
    SUPPORTED_LANGUAGES,
    get_language_instruction,
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Bodhi AI Tutor",
    page_icon="📚",
    layout="wide",
)


# ============================================================================
# SESSION STATE
# ============================================================================

def init_state() -> None:
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "Telugu"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "document_name" not in st.session_state:
        st.session_state.document_name = None

    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = 0

    if "document_pages" not in st.session_state:
        st.session_state.document_pages = 0

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None

    # Teach-back state
    if "show_teach_back" not in st.session_state:
        st.session_state.show_teach_back = False

    if "teach_back_question" not in st.session_state:
        st.session_state.teach_back_question = None

    if "teach_back_chunks" not in st.session_state:
        st.session_state.teach_back_chunks = []

    if "teach_back_language" not in st.session_state:
        st.session_state.teach_back_language = None

    if "teach_back_result" not in st.session_state:
        st.session_state.teach_back_result = None

    if "teach_back_answer" not in st.session_state:
        st.session_state.teach_back_answer = ""

        # Adaptive Practice state

    if "practice_data" not in st.session_state:
        st.session_state.practice_data = None

    if "practice_difficulty" not in st.session_state:
        st.session_state.practice_difficulty = "Medium"

    if "practice_result" not in st.session_state:
        st.session_state.practice_result = None

    if "practice_score" not in st.session_state:
        st.session_state.practice_score = None

    if "practice_answers" not in st.session_state:
        st.session_state.practice_answers = {}

    if "practice_short_result" not in st.session_state:
        st.session_state.practice_short_result = None


init_state()


# ============================================================================
# SOURCE DISPLAY
# ============================================================================

def render_sources(chunks) -> None:

    with st.expander(
        f"📚 Sources ({len(chunks)} retrieved chunks)"
    ):

        for i, chunk in enumerate(chunks, 1):

            page = chunk.metadata.get(
                "page",
                "?",
            )

            st.markdown(
                f"**Chunk {i}** · `page {page}`"
            )

            st.code(
                chunk.page_content,
                language="text",
            )


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_uploaded_pdf(
    uploaded_file,
) -> None:
    """Save and index the uploaded PDF."""

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="bodhi_"
        )
    )

    pdf_path = (
        temp_dir
        / uploaded_file.name
    )

    with open(
        pdf_path,
        "wb",
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    chroma_dir = (
        temp_dir
        / "chroma_db"
    )

    collection_name = "textbook_upload"

    with st.status(
        "Processing your textbook...",
        expanded=True,
    ) as status:

        st.write(
            "📄 Reading the uploaded PDF..."
        )

        try:

            pages = ingest.load_pdf(
                pdf_path
            )

            if not pages:

                raise ValueError(
                    "No readable text was found "
                    "in this PDF."
                )

            st.write(
                f"✅ Extracted {len(pages)} pages."
            )

            st.write(
                "✂️ Splitting textbook into chunks..."
            )

            chunks = ingest.chunk_documents(
                pages
            )

            if not chunks:

                raise ValueError(
                    "The PDF produced no usable chunks."
                )

            st.write(
                f"✅ Created {len(chunks)} chunks."
            )

            st.write(
                "🧠 Creating embeddings..."
            )

            vectorstore = ingest.build_vectorstore(
                chunks,
                persist_directory=chroma_dir,
                collection_name=collection_name,
            )

            count = (
                vectorstore._collection.count()
            )

            st.write(
                f"✅ Stored {count} textbook chunks."
            )

            st.session_state.vectorstore = vectorstore

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.document_chunks = count

            st.session_state.document_pages = (
                len(pages)
            )

            st.session_state.messages = []

            st.session_state.temp_dir = str(
                temp_dir
            )

            # Reset Teach-back state.
            st.session_state.show_teach_back = False
            st.session_state.teach_back_question = None
            st.session_state.teach_back_chunks = []
            st.session_state.teach_back_language = None
            st.session_state.teach_back_result = None
            st.session_state.teach_back_answer = ""

            status.update(
                label="✅ Textbook ready!",
                state="complete",
            )

        except Exception as exc:

            status.update(
                label="❌ Could not process textbook",
                state="error",
            )

            st.error(
                f"Document processing failed: {exc}"
            )

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


# ============================================================================
# TEACH-BACK STATE
# ============================================================================

def open_teach_back(
    question: str,
    chunks,
    language: str,
) -> None:
    """Open the Teach-back panel."""

    st.session_state.teach_back_question = (
        question
    )

    st.session_state.teach_back_chunks = (
        chunks
    )

    st.session_state.teach_back_language = (
        language
    )

    st.session_state.teach_back_result = None

    st.session_state.teach_back_answer = ""

    st.session_state.show_teach_back = True


# ============================================================================
# TEACH-BACK EVALUATION
# ============================================================================

def run_teach_back_evaluation(
    student_answer: str,
) -> None:
    """Evaluate the student's explanation using textbook context."""

    question = (
        st.session_state.teach_back_question
    )

    chunks = (
        st.session_state.teach_back_chunks
    )

    language = (
        st.session_state.teach_back_language
    )

    if not question or not chunks:

        st.error(
            "No textbook explanation is available "
            "for Teach-back."
        )

        return

    if not student_answer.strip():

        st.warning(
            "Please explain the concept in your own words first."
        )

        return

    language_instruction = (
        get_language_instruction(
            st.session_state.selected_language
        )
    )

    with st.spinner(
        "🧠 Checking your explanation..."
    ):

        try:

            result = llm.evaluate_teach_back(
                question=question,
                chunks=chunks,
                student_answer=student_answer,
                language_instruction=language_instruction,
            )

        except (
            APIConnectionError,
            RateLimitError,
            APIError,
        ) as exc:

            st.error(
                f"LLM API call failed: {exc}"
            )

            return

        except RuntimeError as exc:

            st.error(
                str(exc)
            )

            return

    st.session_state.teach_back_result = result

def generate_practice_questions() -> None:
    """Generate adaptive practice questions."""

    if st.session_state.vectorstore is None:
        st.error(
            "Please upload a textbook first."
        )
        return

    language_instruction = (
        get_language_instruction(
            st.session_state.selected_language
        )
    )

    raw = st.session_state.vectorstore.get(
        include=["documents", "metadatas"],
        limit=3,
    )

    from langchain_core.documents import Document

    chunks = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(raw["documents"], raw["metadatas"])
    ]
    

    if len(chunks) == 0:
        st.error("No textbook chunks available.")
        return

    with st.spinner(
        "🧠 Creating practice questions..."
    ):

        try:

            result = llm.generate_practice(
                chunks=chunks,
                language_instruction=language_instruction,
                difficulty=(
                    st.session_state.practice_difficulty
                ),
            )

        except (
            APIConnectionError,
            RateLimitError,
            APIError,
        ) as exc:

            st.error(
                f"LLM API call failed: {exc}"
            )
            return

        except RuntimeError as exc:

            st.error(
                str(exc)
            )
            return

    st.session_state.practice_data = result.data
    st.session_state.practice_result = result
    st.session_state.practice_score = None
    st.session_state.practice_answers = {}
    st.session_state.practice_short_result = None


# ============================================================================
# TEACH-BACK UI
# ============================================================================

def render_teach_back() -> None:
    """Render the separate Teach-back textbox."""

    if not st.session_state.show_teach_back:
        return

    question = (
        st.session_state.teach_back_question
    )

    language = (
        st.session_state.teach_back_language
    )

    st.divider()

    st.subheader(
        "🧠 Teach it back"
    )

    st.write(
        "Now explain the concept in your own words."
    )

    st.info(
        "Don't copy Bodhi's answer. "
        "Explain what you understood."
    )

    st.markdown(
        f"**Concept:** {question}"
    )

    # This is a completely separate textbox.
    student_answer = st.text_area(
        "Your explanation",
        placeholder=(
            "Explain what you understood in your own words..."
        ),
        height=180,
        key="teach_back_textbox",
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔍 Check my explanation",
            key="check_teach_back",
            use_container_width=True,
        ):

            st.session_state.teach_back_answer = (
                student_answer
            )

            run_teach_back_evaluation(
                student_answer
            )

    with col2:

        if st.button(
            "✖️ Close Teach-back",
            key="close_teach_back",
            use_container_width=True,
        ):

            st.session_state.show_teach_back = False
            st.session_state.teach_back_result = None
            st.session_state.teach_back_question = None
            st.session_state.teach_back_chunks = []
            st.session_state.teach_back_language = None
            st.session_state.teach_back_answer = ""

            st.rerun()

    result = (
        st.session_state.teach_back_result
    )

    if result is not None:

        st.divider()

        st.subheader(
            "📊 Your Teach-back Feedback"
        )

        st.markdown(
            result.text
        )

        st.caption(
            f"Evaluation time: "
            f"{result.latency_seconds:.2f} seconds · "
            f"Language: {language}"
        )


# ============================================================================
# ANSWER GENERATION
# ============================================================================

def answer_question(
    question: str,
    vectorstore,
    language: str,
) -> None:
    """Retrieve textbook context and generate a grounded answer."""

    # ------------------------------------------------------------------------
    # Retrieve relevant textbook chunks.
    # ------------------------------------------------------------------------

    chunks = rag.retrieve(
        question,
        vectorstore,
    )

    # ------------------------------------------------------------------------
    # HARD GROUNDING GATE
    # ------------------------------------------------------------------------

    if not chunks:

        refusal = (
            "I couldn't find this information in the uploaded "
            "textbook. Please ask a question about the content "
            "you uploaded."
        )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": refusal,
                "latency": 0.0,
                "chunks": [],
                "question": question,
                "language": language,
            }
        )

        st.rerun()

        return

    # ------------------------------------------------------------------------
    # Language instruction.
    # ------------------------------------------------------------------------

    language_instruction = (
        get_language_instruction(
            language
        )
    )

    # ------------------------------------------------------------------------
    # Generate answer.
    # ------------------------------------------------------------------------

    try:

        start = time.perf_counter()

        answer_text = ""

        for delta in llm.stream_answer(
            question,
            chunks,
            language_instruction,
        ):

            answer_text += delta

        elapsed = (
            time.perf_counter()
            - start
        )

    except (
        APIConnectionError,
        RateLimitError,
        APIError,
    ) as exc:

        st.error(
            f"LLM API call failed: {exc}"
        )

        return

    except RuntimeError as exc:

        st.error(
            str(exc)
        )

        return

    # ------------------------------------------------------------------------
    # SAVE BOTH QUESTION AND ANSWER BEFORE RERUN.
    # ------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "latency": elapsed,
            "chunks": chunks,
            "question": question,
            "language": language,
        }
    )

    # IMPORTANT:
    # The answer is now safely stored.
    # On rerun the chat-history section will display it and
    # create its Teach-back button.
    st.rerun()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.header(
        "📚 Bodhi AI Tutor"
    )

    st.write(
        "Upload your own textbook and "
        "learn it in a language you understand."
    )

    st.subheader(
        "📄 Your textbook"
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF textbook or chapter",
        type=["pdf"],
        help=(
            "Upload a textbook, chapter, "
            "or textbook page as a PDF."
        ),
    )

    if uploaded_file is not None:

        current_name = (
            st.session_state.document_name
        )

        if current_name != uploaded_file.name:

            if st.button(
                "🚀 Process this textbook",
                use_container_width=True,
            ):

                process_uploaded_pdf(
                    uploaded_file
                )

    if st.session_state.vectorstore is not None:

        st.success(
            f"Ready: "
            f"{st.session_state.document_name}"
        )

        st.caption(
            f"{st.session_state.document_pages} pages · "
            f"{st.session_state.document_chunks} chunks"
        )

    st.divider()

    st.subheader(
        "🌐 Teaching language"
    )

    language = st.selectbox(
        "Choose your teaching language",
        list(
            SUPPORTED_LANGUAGES.keys()
        ),
        index=list(
            SUPPORTED_LANGUAGES.keys()
        ).index("Telugu"),
        key='selected_language',
    )

    st.divider()

    st.subheader(
        "⚙️ System"
    )

    st.markdown(
        f"- **Embeddings:** "
        f"`{EMBEDDING_MODEL.split('/')[-1]}`\n"
        f"- **LLM:** "
        f"`{LLM_MODEL.split('/')[-1]}`\n"
        f"- **Chunking:** "
        f"{CHUNK_SIZE} / {CHUNK_OVERLAP}\n"
        f"- **Retrieval:** "
        f"top-{TOP_K}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.show_teach_back = False
        st.session_state.teach_back_question = None
        st.session_state.teach_back_chunks = []
        st.session_state.teach_back_language = None
        st.session_state.teach_back_result = None
        st.session_state.teach_back_answer = ""

        st.rerun()


# ============================================================================
# MAIN UI
# ============================================================================

st.title(
    "📚 Bodhi AI Tutor"
)

st.caption(
    "Upload your textbook. "
    "Ask questions. "
    "Learn in your mother tongue."
)


if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload a textbook or chapter "
        "from the sidebar to start learning."
    )

    st.stop()


# ============================================================================
# DOCUMENT INFORMATION
# ============================================================================

st.success(
    f"📖 **{st.session_state.document_name}** "
    f"is ready."
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Pages",
        st.session_state.document_pages,
    )

with col2:

    st.metric(
        "Chunks",
        st.session_state.document_chunks,
    )

with col3:

    st.metric(
        "Language",
        language,
    )

# ============================================================================
# ADAPTIVE PRACTICE
# ============================================================================

# ============================================================================
# ADAPTIVE PRACTICE
# ============================================================================

st.divider()

st.subheader("📘 Adaptive Practice")

st.write(
    f"Current difficulty: "
    f"**{st.session_state.practice_difficulty}**"
)

if st.button(
    "🚀 Generate Practice",
    use_container_width=True,
):
    generate_practice_questions()


if st.session_state.practice_data:

    st.success("Quiz generated!")

    for i, mcq in enumerate(
        st.session_state.practice_data["mcqs"],
        start=1
    ):

        st.write(
            f"### Q{i}. {mcq['question']}"
        )

        for option in mcq["options"]:

            st.write(
                f"{option['label']}. {option['text']}"
            )

        st.info(
            f"Correct Answer: {mcq['correct_answer']}"
        )

# ============================================================================
# CHAT HISTORY
# ============================================================================
#
# Every assistant answer gets its own Teach-back button here.
# This includes the FIRST answer.
# ============================================================================

for index, msg in enumerate(
    st.session_state.messages
):

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

        if msg["role"] == "assistant":

            st.caption(
                f"{msg['latency']:.2f} s · "
                f"top-{len(msg['chunks'])} retrieval"
            )

            if msg["chunks"]:

                render_sources(
                    msg["chunks"]
                )

                saved_question = msg.get(
                    "question"
                )

                saved_language = msg.get(
                    "language",
                    language,
                )

                if saved_question:

                    if st.button(
                        "🧠 Teach it back",
                        key=f"teach_back_history_{index}",
                        use_container_width=True,
                    ):

                        open_teach_back(
                            saved_question,
                            msg["chunks"],
                            saved_language,
                        )

                        st.rerun()


# ============================================================================
# TEACH-BACK PANEL
# ============================================================================

render_teach_back()


# ============================================================================
# NORMAL CHAT INPUT
# ============================================================================

question = st.chat_input(
    "Ask about your uploaded textbook..."
)


if st.session_state.pending_question:

    question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None


if question and question.strip():

    answer_question(
        question.strip(),
        st.session_state.vectorstore,
        language,
    )