from pathlib import Path
import hashlib
import json
import re


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\Desktop\AI_Document_Intelligence")

# Correct input file location
INPUT_FILE = (
    PROJECT_DIR
    / "processed_data"
    / "extracted_documents.json"
)

# Output location
OUTPUT_DIR = PROJECT_DIR / "processed_data"
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

# Chunk configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_configuration():
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative.")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )


# ============================================================
# FIND INPUT FILE
# ============================================================

def find_input_file():
    """
    Find extracted_documents.json in the expected location.
    If it is not there, check common alternative locations.
    """

    possible_paths = [
        INPUT_FILE,
        PROJECT_DIR / "extracted_documents.json",
        PROJECT_DIR / "processed_data" / "processed_data"
        / "extracted_documents.json",
    ]

    for path in possible_paths:
        if path.is_file():
            print(f"Input file found: {path}")
            return path

    # Search the project if the file is in another subfolder.
    for path in PROJECT_DIR.rglob("extracted_documents.json"):
        if path.is_file():
            print(f"Input file found in alternative location: {path}")
            return path

    raise FileNotFoundError(
        "\nCould not find extracted_documents.json.\n"
        f"Expected location: {INPUT_FILE}\n\n"
        "Run document_processor.py first and verify that "
        "the JSON output was created."
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    """Normalize whitespace in extracted text."""

    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# GENERATE CHUNK ID
# ============================================================

def generate_chunk_id(document_id, page_number, chunk_index, text):
    """Generate a deterministic identifier for each chunk."""

    content = (
        f"{document_id}|{page_number}|"
        f"{chunk_index}|{text}"
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()[:20]


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def split_into_chunks(text):
    """
    Split text into word-based chunks with overlap.
    """

    text = clean_text(text)

    if not text:
        return []

    words = text.split()
    chunks = []

    start = 0
    total_words = len(words)

    while start < total_words:
        end = min(start + CHUNK_SIZE, total_words)

        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= total_words:
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    """Load extracted documents from JSON."""

    input_path = find_input_file()

    try:
        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as file:
            documents = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON file: {error}"
        ) from error

    if not isinstance(documents, list):
        raise ValueError(
            "Expected a JSON list containing extracted documents."
        )

    if not documents:
        raise ValueError(
            "The JSON file contains no documents."
        )

    return documents


# ============================================================
# CHUNK ONE DOCUMENT
# ============================================================

def chunk_document(document):
    """Create chunks from every extracted page in a document."""

    if not isinstance(document, dict):
        return []

    file_name = document.get("file_name", "unknown.pdf")

    document_id = document.get("document_id")

    if not document_id:
        document_id = hashlib.sha256(
            file_name.encode("utf-8")
        ).hexdigest()[:16]

    document_metadata = document.get(
        "document_metadata", {}
    )

    if not isinstance(document_metadata, dict):
        document_metadata = {}

    pages = document.get("pages", [])

    if not isinstance(pages, list):
        print(f"WARNING: Invalid page records in {file_name}")
        return []

    document_chunks = []

    for page in pages:
        if not isinstance(page, dict):
            continue

        text = clean_text(page.get("text", ""))

        if not text:
            continue

        page_number = page.get("page_number", 0)

        page_metadata = page.get("metadata", {})

        if not isinstance(page_metadata, dict):
            page_metadata = {}

        chunks = split_into_chunks(text)

        for chunk_index, chunk_text in enumerate(
            chunks,
            start=1
        ):
            chunk_id = generate_chunk_id(
                document_id,
                page_number,
                chunk_index,
                chunk_text
            )

            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "file_name": file_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "metadata": {
                    "document_title": document_metadata.get(
                        "title", ""
                    ),
                    "document_author": document_metadata.get(
                        "author", ""
                    ),
                    "document_subject": document_metadata.get(
                        "subject", ""
                    ),
                    "file_path": page_metadata.get(
                        "file_path",
                        document.get("file_path", "")
                    ),
                    "file_hash": page_metadata.get(
                        "file_hash",
                        document.get("file_hash", "")
                    ),
                    "extraction_method": page.get(
                        "extraction_method", "unknown"
                    ),
                    "page_width": page_metadata.get(
                        "page_width"
                    ),
                    "page_height": page_metadata.get(
                        "page_height"
                    )
                }
            }

            document_chunks.append(chunk_record)

    return document_chunks


# ============================================================
# PROCESS ALL DOCUMENTS
# ============================================================

def create_all_chunks(documents):
    """Process every successfully extracted document."""

    all_chunks = []

    print("\n" + "=" * 65)
    print("AI DOCUMENT INTELLIGENCE - TEXT CHUNKING")
    print("=" * 65)

    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Chunk overlap: {CHUNK_OVERLAP} words")
    print(f"Documents loaded: {len(documents)}")

    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            print(f"\n[{index}] Invalid document record. Skipping.")
            continue

        file_name = document.get("file_name", "unknown.pdf")
        status = document.get("status", "success")

        print(f"\n[{index}/{len(documents)}] {file_name}")

        if status == "failed":
            print("Skipped because extraction failed.")
            continue

        chunks = chunk_document(document)
        all_chunks.extend(chunks)

        print(f"Chunks created: {len(chunks)}")

    return all_chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(chunks):
    """Save chunks as JSON."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nOutput saved to: {OUTPUT_FILE}")


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(documents, chunks):
    """Display chunking statistics."""

    total_words = sum(
        chunk["word_count"]
        for chunk in chunks
    )

    unique_documents = len({
        chunk["document_id"]
        for chunk in chunks
    })

    print("\n" + "=" * 65)
    print("CHUNKING SUMMARY")
    print("=" * 65)

    print(f"Documents loaded:        {len(documents)}")
    print(f"Documents with chunks:   {unique_documents}")
    print(f"Total chunks:            {len(chunks)}")
    print(f"Total words in chunks:   {total_words}")

    if chunks:
        average_words = total_words / len(chunks)

        print(
            f"Average words per chunk: {average_words:.2f}"
        )

        first_chunk = chunks[0]

        print("\nFirst chunk preview:")
        print(f"Chunk ID: {first_chunk['chunk_id']}")
        print(f"Document: {first_chunk['file_name']}")
        print(f"Page: {first_chunk['page_number']}")
        print(f"Word count: {first_chunk['word_count']}")
        print(f"Text: {first_chunk['text'][:300]}")

    else:
        print(
            "\nWARNING: No chunks were created. "
            "Check the extracted JSON structure."
        )

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        validate_configuration()

        documents = load_documents()

        chunks = create_all_chunks(documents)

        save_chunks(chunks)

        print_summary(documents, chunks)

    except Exception as error:
        print("\nCHUNKING PIPELINE ERROR")
        print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()