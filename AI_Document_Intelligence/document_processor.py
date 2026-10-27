from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pymupdf
import pytesseract
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "sample_documents"
OUTPUT_DIR = PROJECT_DIR / "processed_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "extracted_documents.json"

# Default Windows installation path for Tesseract.
TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# OCR settings.
OCR_LANGUAGE = "eng"
OCR_DPI = 250

# If a page contains fewer than this many characters,
# try OCR as a fallback.
MIN_TEXT_CHARACTERS = 30


# ============================================================
# INITIALIZE OCR
# ============================================================

def initialize_ocr():
    """Configure Tesseract OCR."""

    if not TESSERACT_PATH.exists():
        raise FileNotFoundError(
            "Tesseract OCR was not found.\n"
            f"Expected location: {TESSERACT_PATH}\n"
            "Install Tesseract or update TESSERACT_PATH."
        )

    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_PATH
    )

    # Check that Tesseract can execute.
    version = pytesseract.get_tesseract_version()

    print(f"Tesseract OCR version: {version}")


# ============================================================
# FILE IDENTIFICATION
# ============================================================

def calculate_file_hash(file_path):
    """Calculate a SHA-256 hash for a PDF."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


def generate_document_id(file_path, file_hash):
    """Generate a stable ID for a document."""

    unique_value = f"{file_path.name}:{file_hash}"

    return hashlib.sha256(
        unique_value.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# OCR A SINGLE PDF PAGE
# ============================================================

def extract_text_with_ocr(page):
    """
    Render a PDF page as an image and recognize its text
    using Tesseract OCR.
    """

    pixmap = page.get_pixmap(
        dpi=OCR_DPI,
        colorspace=pymupdf.csRGB,
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples
    )

    recognized_text = pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGE,
        config="--psm 3"
    )

    return recognized_text.strip()


# ============================================================
# PROCESS A SINGLE PDF
# ============================================================

def process_pdf(file_path):
    """Extract text from a PDF, using OCR when necessary."""

    file_path = Path(file_path)

    result = {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "document_id": None,
        "file_hash": None,
        "document_metadata": {},
        "total_pages": 0,
        "processed_pages": 0,
        "ocr_pages": 0,
        "skipped_empty_pages": 0,
        "failed_pages": [],
        "pages": [],
        "status": "failed",
        "error": None,
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    try:
        # ----------------------------------------------------
        # Validate file
        # ----------------------------------------------------

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                "The supplied path is not a file."
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                "The supplied file is not a PDF."
            )

        if file_path.stat().st_size == 0:
            raise ValueError(
                "The PDF file is empty."
            )

        result["file_hash"] = calculate_file_hash(
            file_path
        )

        result["document_id"] = generate_document_id(
            file_path,
            result["file_hash"]
        )

        # ----------------------------------------------------
        # Open PDF
        # ----------------------------------------------------

        with pymupdf.open(file_path) as pdf:

            if pdf.needs_pass:
                result["error"] = (
                    "Password-protected PDF. "
                    "An authorized, accessible copy is required."
                )
                return result

            result["total_pages"] = len(pdf)

            metadata = pdf.metadata or {}

            result["document_metadata"] = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get(
                    "creationDate", ""
                ),
                "modification_date": metadata.get(
                    "modDate", ""
                )
            }

            if len(pdf) == 0:
                result["error"] = (
                    "The PDF contains no pages."
                )
                return result

            # ------------------------------------------------
            # Process each page
            # ------------------------------------------------

            for page_number, page in enumerate(
                pdf,
                start=1
            ):

                try:
                    # First attempt normal PDF text extraction.
                    text = page.get_text("text").strip()

                    extraction_method = "text"

                    # If little or no text is available,
                    # try OCR on the rendered page.
                    if len(text) < MIN_TEXT_CHARACTERS:

                        try:
                            ocr_text = extract_text_with_ocr(
                                page
                            )

                            if len(ocr_text) > len(text):
                                text = ocr_text
                                extraction_method = "ocr"

                        except Exception as ocr_error:
                            result["failed_pages"].append({
                                "page_number": page_number,
                                "stage": "ocr",
                                "error": (
                                    f"{type(ocr_error).__name__}: "
                                    f"{ocr_error}"
                                )
                            })

                            print(
                                f"OCR warning on "
                                f"{file_path.name}, "
                                f"page {page_number}: "
                                f"{ocr_error}"
                            )

                    # Skip pages where neither method
                    # extracted useful text.
                    if not text.strip():
                        result["skipped_empty_pages"] += 1
                        continue

                    # Store extracted page text and metadata.
                    page_record = {
                        "document_id": result["document_id"],
                        "file_name": file_path.name,
                        "page_number": page_number,
                        "text": text,
                        "extraction_method": extraction_method,
                        "metadata": {
                            "file_path": str(
                                file_path.resolve()
                            ),
                            "file_hash": result["file_hash"],
                            "document_title": (
                                result["document_metadata"]
                                .get("title", "")
                            ),
                            "page_number": page_number,
                            "page_width": page.rect.width,
                            "page_height": page.rect.height
                        }
                    }

                    result["pages"].append(page_record)
                    result["processed_pages"] += 1

                    if extraction_method == "ocr":
                        result["ocr_pages"] += 1

                    print(
                        f"  Page {page_number}/"
                        f"{result['total_pages']}: "
                        f"{extraction_method.upper()}"
                    )

                except Exception as page_error:

                    result["failed_pages"].append({
                        "page_number": page_number,
                        "stage": "page_processing",
                        "error": (
                            f"{type(page_error).__name__}: "
                            f"{page_error}"
                        )
                    })

            # ------------------------------------------------
            # Determine final status
            # ------------------------------------------------

            if result["processed_pages"] > 0:

                if result["failed_pages"]:
                    result["status"] = "partial_success"
                else:
                    result["status"] = "success"

            else:
                result["status"] = "failed"

                result["error"] = (
                    "No text was extracted. The PDF may be "
                    "empty, image-only with OCR failure, "
                    "or contain unreadable content."
                )

    except Exception as error:

        result["status"] = "failed"

        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return result


# ============================================================
# PROCESS MULTIPLE PDF FILES
# ============================================================

def process_all_pdfs(folder_path=DOCUMENTS_DIR):
    """Process all PDFs in a folder."""

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {folder_path}"
        )

    pdf_files = sorted(
        [
            file
            for file in folder_path.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".pdf"
        ],
        key=lambda file: file.name.lower()
    )

    if not pdf_files:
        print(
            f"No PDFs found in {folder_path}"
        )
        return []

    results = []

    print("\n" + "=" * 65)
    print("AI DOCUMENT INTELLIGENCE - OCR PIPELINE")
    print("=" * 65)

    print(f"PDF files found: {len(pdf_files)}")

    for index, pdf_file in enumerate(
        pdf_files,
        start=1
    ):

        print("\n" + "-" * 65)
        print(
            f"[{index}/{len(pdf_files)}] "
            f"Processing: {pdf_file.name}"
        )

        result = process_pdf(pdf_file)

        results.append(result)

        print(f"Status: {result['status']}")
        print(
            f"Extracted pages: "
            f"{result['processed_pages']}/"
            f"{result['total_pages']}"
        )
        print(
            f"Pages processed using OCR: "
            f"{result['ocr_pages']}"
        )
        print(
            f"Empty pages skipped: "
            f"{result['skipped_empty_pages']}"
        )

        if result["error"]:
            print(f"Error: {result['error']}")

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):
    """Save extracted text and metadata to JSON."""

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        json.dump(
            results,
            output_file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(results):

    total_files = len(results)

    successful_files = sum(
        item["status"] == "success"
        for item in results
    )

    partial_files = sum(
        item["status"] == "partial_success"
        for item in results
    )

    failed_files = sum(
        item["status"] == "failed"
        for item in results
    )

    total_pages = sum(
        item["total_pages"]
        for item in results
    )

    extracted_pages = sum(
        item["processed_pages"]
        for item in results
    )

    ocr_pages = sum(
        item["ocr_pages"]
        for item in results
    )

    empty_pages = sum(
        item["skipped_empty_pages"]
        for item in results
    )

    print("\n" + "=" * 65)
    print("PROCESSING SUMMARY")
    print("=" * 65)

    print(f"Total PDFs:              {total_files}")
    print(f"Successful PDFs:         {successful_files}")
    print(f"Partially successful:    {partial_files}")
    print(f"Failed PDFs:             {failed_files}")
    print(f"Total pages:             {total_pages}")
    print(f"Pages with extracted text: {extracted_pages}")
    print(f"Pages processed using OCR: {ocr_pages}")
    print(f"Empty pages skipped:     {empty_pages}")

    for item in results:

        if item["status"] != "success":

            print(
                f"\nNeeds attention: "
                f"{item['file_name']}"
            )

            if item["error"]:
                print(f"Reason: {item['error']}")

            for page_error in item["failed_pages"]:
                print(
                    f"Page {page_error['page_number']}: "
                    f"{page_error['error']}"
                )

    print("=" * 65)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    try:
        initialize_ocr()

        results = process_all_pdfs()

        if not results:
            print(
                "\nPlace PDF files inside:\n"
                f"{DOCUMENTS_DIR}"
            )
            return

        save_results(results)
        print_summary(results)

    except Exception as error:

        print("\nPIPELINE ERROR")

        print(
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()