import hashlib
import io
import time

import fitz
import streamlit as st

from retrieval_engine import RetrievalEngine
from rag_pipeline import RAGPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI-Powered Document Intelligence")

st.write(
    "Upload PDF documents and ask questions using hybrid retrieval, "
    "cross-encoder reranking, and retrieval-augmented generation."
)


# ============================================================
# SESSION STATE
# ============================================================

if "retrieval_engine" not in st.session_state:
    st.session_state.retrieval_engine = None

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_processed_signature" not in st.session_state:
    st.session_state.last_processed_signature = None


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def split_page_into_chunks(
    text,
    chunk_size=1800,
    overlap=250
):
    """
    Split page text into character-based chunks.

    These settings are characters, not words.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk_text = text[start:start + chunk_size].strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_uploaded_pdfs(uploaded_files):
    """
    Extract selectable text from uploaded PDF files.

    Each chunk retains its original filename and page number.
    """

    all_chunks = []
    document_names = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        document_id = file_hash[:16]

        document_names.append(uploaded_file.name)

        with fitz.open(
            stream=io.BytesIO(pdf_bytes),
            filetype="pdf"
        ) as pdf:

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                page_chunks = split_page_into_chunks(text)

                for chunk_index, chunk_text in enumerate(
                    page_chunks,
                    start=1
                ):
                    chunk_id = hashlib.sha256(
                        (
                            f"{file_hash}:"
                            f"{page_number}:"
                            f"{chunk_index}:"
                            f"{chunk_text}"
                        ).encode("utf-8")
                    ).hexdigest()[:20]

                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "file_name": uploaded_file.name,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "metadata": {
                            "file_hash": file_hash
                        }
                    })

    return all_chunks, list(dict.fromkeys(document_names))


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button(
    "Process Documents",
    type="primary"
):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:
        signature_data = []

        for uploaded_file in uploaded_files:
            signature_data.append({
                "name": uploaded_file.name,
                "hash": hashlib.sha256(
                    uploaded_file.getvalue()
                ).hexdigest()
            })

        signature = hashlib.sha256(
            str(signature_data).encode("utf-8")
        ).hexdigest()

        try:
            with st.spinner(
                "Extracting PDF text and preparing document chunks..."
            ):
                chunks, document_names = extract_uploaded_pdfs(
                    uploaded_files
                )

            if not chunks:
                st.error(
                    "No selectable text was found in these PDFs. "
                    "Scanned documents require OCR."
                )

            else:
                engine = RetrievalEngine(
                    enable_reranking=True
                )

                with st.spinner(
                    "Loading AI models and building hybrid search indexes..."
                ):
                    engine.set_chunks(chunks)
                    engine.build_indexes()

                st.session_state.retrieval_engine = engine
                st.session_state.rag_pipeline = RAGPipeline()
                st.session_state.document_names = document_names
                st.session_state.chat_history = []
                st.session_state.last_processed_signature = signature

                st.success(
                    f"Successfully processed {len(document_names)} "
                    f"document(s) into {len(chunks)} chunks."
                )

        except Exception as error:
            st.error(
                f"Document processing failed: "
                f"{type(error).__name__}: {error}"
            )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.retrieval_engine is not None:

    st.divider()

    st.subheader("📄 Processed Documents")

    for document_name in st.session_state.document_names:
        st.write(f"✅ {document_name}")

    st.caption(
        f"Indexed chunks: "
        f"{len(st.session_state.retrieval_engine.chunks)}"
    )


# ============================================================
# QUESTION AND ANSWER
# ============================================================

if st.session_state.retrieval_engine is not None:

    st.divider()

    st.subheader("💬 Ask Questions About Your Documents")

    question = st.text_input(
        "Enter your question",
        key="document_question",
        placeholder="Example: What are the main findings of this report?"
    )

    top_k = st.slider(
        "Number of source passages",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Get Answer", type="primary"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                engine = st.session_state.retrieval_engine
                pipeline = st.session_state.rag_pipeline

                with st.spinner(
                    "Searching documents and generating an answer..."
                ):
                    retrieval_result = engine.retrieve(
                        query=question,
                        top_k=top_k
                    )

                    retrieved_chunks = retrieval_result["results"]

                    result = pipeline.answer_question(
                        question,
                        retrieved_chunks
                    )

                st.subheader("🧠 Answer")

                st.markdown(result["answer"])

                st.caption(
                    "Retrieval time: "
                    f"{retrieval_result['retrieval_time_seconds']:.4f} seconds"
                )

                st.subheader("📚 Retrieved Sources")

                if not result["sources"]:
                    st.info("No relevant source passages were found.")

                for source in result["sources"]:
                    with st.expander(
                        f"{source['source']} — Page {source['page']}"
                    ):
                        st.write(
                            f"Retrieval score: {source['score']:.4f}"
                        )

                        matching_chunks = [
                            chunk
                            for chunk in retrieved_chunks
                            if chunk["source"] == source["source"]
                            and chunk["page"] == source["page"]
                        ]

                        for chunk in matching_chunks:
                            st.write(chunk["text"])

                            if chunk.get("semantic_score") is not None:
                                st.caption(
                                    "Semantic score: "
                                    f"{chunk['semantic_score']:.4f}"
                                )

                            if chunk.get("keyword_score") is not None:
                                st.caption(
                                    "BM25 score: "
                                    f"{chunk['keyword_score']:.4f}"
                                )

                            if chunk.get("rerank_score") is not None:
                                st.caption(
                                    "Cross-encoder score: "
                                    f"{chunk['rerank_score']:.4f}"
                                )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"]
                })

            except Exception as error:
                st.error(
                    f"Unable to answer the question: "
                    f"{type(error).__name__}: {error}"
                )


# ============================================================
# RECENT QUESTION HISTORY
# ============================================================

if st.session_state.chat_history:

    st.divider()

    st.subheader("🕘 Recent Questions")

    for index, item in enumerate(
        reversed(st.session_state.chat_history),
        start=1
    ):
        with st.expander(
            f"{index}. {item['question']}"
        ):
            st.markdown(item["answer"])

            st.write("Sources:")

            for source in item["sources"]:
                st.write(
                    f"- {source['source']}, page {source['page']}"
                )