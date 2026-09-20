"""PDF Processor Module.

Handles validation, page inspection, and text extraction from PDF documents using pypdf.
"""

import io
from typing import Dict, Any, List, Tuple
from pypdf import PdfReader


def validate_and_extract_pdf(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Validate and extract textual content from uploaded PDF bytes.

    Args:
        file_bytes: Raw binary bytes of the uploaded PDF file.
        filename: Original file name.

    Returns:
        dict containing:
            - success (bool): True if extraction succeeded.
            - error (str): Error description if extraction failed.
            - filename (str): Document file name.
            - file_size_bytes (int): Total size in bytes.
            - page_count (int): Total number of pages.
            - pages_text (List[Tuple[int, str]]): List of (page_number, extracted_text).
            - full_text (str): Combined raw text from all pages.
    """
    if not file_bytes:
        return {
            "success": False,
            "error": "The uploaded file is empty (0 bytes).",
            "filename": filename,
            "file_size_bytes": 0,
            "page_count": 0,
            "pages_text": [],
            "full_text": "",
        }

    # Verify standard PDF magic bytes header (%PDF)
    if not file_bytes.startswith(b"%PDF"):
        return {
            "success": False,
            "error": "The file does not appear to be a valid PDF format (missing %PDF header).",
            "filename": filename,
            "file_size_bytes": len(file_bytes),
            "page_count": 0,
            "pages_text": [],
            "full_text": "",
        }

    try:
        stream = io.BytesIO(file_bytes)
        reader = PdfReader(stream)
        page_count = len(reader.pages)

        if page_count == 0:
            return {
                "success": False,
                "error": "The PDF file contains 0 pages.",
                "filename": filename,
                "file_size_bytes": len(file_bytes),
                "page_count": 0,
                "pages_text": [],
                "full_text": "",
            }

        pages_text: List[Tuple[int, str]] = []
        full_text_parts: List[str] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages_text.append((page_idx, text))
            if text.strip():
                full_text_parts.append(text.strip())

        full_text = "\n\n".join(full_text_parts)

        # Check if any extractable text was found
        if not full_text.strip():
            return {
                "success": False,
                "error": (
                    "No extractable text was found in the PDF. "
                    "The document may contain scanned images without an OCR text layer."
                ),
                "filename": filename,
                "file_size_bytes": len(file_bytes),
                "page_count": page_count,
                "pages_text": pages_text,
                "full_text": "",
            }

        return {
            "success": True,
            "error": "",
            "filename": filename,
            "file_size_bytes": len(file_bytes),
            "page_count": page_count,
            "pages_text": pages_text,
            "full_text": full_text,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse PDF document: {str(e)}",
            "filename": filename,
            "file_size_bytes": len(file_bytes),
            "page_count": 0,
            "pages_text": [],
            "full_text": "",
        }
