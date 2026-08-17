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


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bodhi AI Tutor",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:

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


init_state()


# ---------------------------------------------------------------------------
# Source display
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def process_uploaded_pdf(
    uploaded_file,
) -> None:
    """Save and index the uploaded PDF."""

    # Create a temporary directory for this document.
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

    # Give each uploaded document its own collection.
    collection_name = (
        "textbook_upload"
    )

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

            chunks = (
                ingest.chunk_documents(
                    pages
                )
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

            vectorstore = (
                ingest.build_vectorstore(
                    chunks,
                    persist_directory=chroma_dir,
                    collection_name=collection_name,
                )
            )

            count = (
                vectorstore._collection.count()
            )

            st.write(
                f"✅ Stored {count} textbook chunks."
            )

            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.document_chunks = (
                count
            )

            st.session_state.document_pages = (
                len(pages)
            )

            st.session_state.messages = []

            st.session_state.temp_dir = (
                str(temp_dir)
            )

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

            # Clean up failed upload.
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    vectorstore,
    language: str,
) -> None:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("assistant"):

        with st.status(
            "Thinking...",
            expanded=True,
        ) as status:

            st.write(
                f"🔎 Searching the uploaded textbook "
                f"for: _{question}_"
            )

            chunks = rag.retrieve(
                question,
                vectorstore,
            )

            st.write(
                f"📚 Retrieved {len(chunks)} "
                "relevant textbook chunks."
            )

            language_instruction = (
                get_language_instruction(
                    language
                )
            )

            status.update(
                label="Generating answer from textbook...",
                state="running",
            )

        placeholder = st.empty()

        try:

            start = time.perf_counter()

            buffer = ""

            for delta in llm.stream_answer(
                question,
                chunks,
                language_instruction,
            ):

                buffer += delta

                placeholder.markdown(
                    buffer + "▌"
                )

            elapsed = (
                time.perf_counter()
                - start
            )

            placeholder.markdown(
                buffer
            )

        except (
            APIConnectionError,
            RateLimitError,
            APIError,
        ) as exc:

            placeholder.error(
                f"LLM API call failed: {exc}"
            )

            return

        except RuntimeError as exc:

            placeholder.error(
                str(exc)
            )

            return

        st.caption(
            f"{elapsed:.2f} s · "
            f"top-{len(chunks)} retrieval · "
            f"language: {language}"
        )

        render_sources(chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": buffer,
            "latency": elapsed,
            "chunks": chunks,
        }
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

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

        st.rerun()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Document information
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Replay chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

        if (
            msg["role"]
            == "assistant"
        ):

            st.caption(
                f"{msg['latency']:.2f} s · "
                f"top-{len(msg['chunks'])} retrieval"
            )

            render_sources(
                msg["chunks"]
            )


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

question = st.chat_input(
    "Ask about your uploaded textbook..."
)


if (
    st.session_state.pending_question
):
    question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = (
        None
    )


if question and question.strip():

    answer_question(
        question.strip(),
        st.session_state.vectorstore,
        language,
    )