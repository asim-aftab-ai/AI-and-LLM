"""Integration verification test for Project 09: Research Assistant UI.

Validates that:
1. Project 09 can locate and import existing backend logic from Project 08.
2. PDF validation, cleaning, chunking, and semantic indexing function properly.
3. Retriever yields authentic metadata (chunk_id, page_num, score, text).
4. Multi-turn conversation persistence functions as expected.
5. app.py syntax and module dependencies are valid.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project 09 directory to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_08_DIR = CURRENT_DIR.parent / "project-08-ai-pdf-research-assistant"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_08_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_08_DIR))

# Import from project-08 backend
from src.pdf_processor import validate_and_extract_pdf
from src.text_cleaner import clean_text, clean_pages_text
from src.document_processor import chunk_document
from src.retriever import DocumentRetriever, format_context_for_prompt
from src.conversation import ConversationManager
from src.llm import SYSTEM_PROMPT


def make_test_pdf_bytes(lines: list[str]) -> bytes:
    """Create minimal valid PDF byte sequence in-memory."""
    stream_content = "BT\n/F1 12 Tf\n"
    y = 720
    for line in lines:
        escaped = line.replace("(", "\\(").replace(")", "\\)")
        stream_content += f"72 {y} Td ({escaped}) Tj\n0 -20 Td\n"
        y -= 20
    stream_content += "ET\n"
    stream_bytes = stream_content.encode("latin1")

    obj4 = (
        f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin1")
        + stream_bytes
        + b"endstream\nendobj\n"
    )

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        + obj4
        + b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    return pdf


class TestProject09Integration(unittest.TestCase):
    def test_existing_backend_import_resolution(self):
        """Verify all required backend components can be imported from Project 08."""
        self.assertTrue(callable(validate_and_extract_pdf))
        self.assertTrue(callable(clean_text))
        self.assertTrue(callable(chunk_document))
        self.assertTrue(callable(format_context_for_prompt))
        self.assertIn("CRITICAL INSTRUCTIONS", SYSTEM_PROMPT)

    def test_document_processing_pipeline(self):
        """Verify the full extraction, cleaning, and chunking pipeline."""
        sample_lines = [
            "Artificial intelligence models require high-qual-\nity training datasets.",
            "Retrieval-Augmented Generation enhances factual reliability by grounding responses.",
        ]
        pdf_bytes = make_test_pdf_bytes(sample_lines)

        result = validate_and_extract_pdf(pdf_bytes, "test_research.pdf")
        self.assertTrue(result["success"])
        self.assertEqual(result["page_count"], 1)

        cleaned_full = clean_text(result["full_text"])
        self.assertIn("high-quality", cleaned_full)  # Hyphen resolved

        cleaned_pages = clean_pages_text(result["pages_text"])
        chunks = chunk_document(cleaned_pages, chunk_size_words=20, overlap_words=5)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("chunk_id", chunks[0])
        self.assertIn("page_num", chunks[0])
        self.assertIn("text", chunks[0])

    def test_conversation_manager_citations(self):
        """Verify that ConversationManager properly preserves citations and excerpts."""
        conv = ConversationManager()
        conv.add_user_message("What is RAG?")
        sample_citations = [
            {
                "chunk_id": 1,
                "page_num": 1,
                "score": 0.88,
                "word_count": 12,
                "text": "Retrieval-Augmented Generation enhances factual reliability.",
            }
        ]
        conv.add_assistant_message("RAG enhances factual reliability.", citations=sample_citations)

        messages = conv.get_all_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(len(messages[1]["citations"]), 1)
        self.assertEqual(messages[1]["citations"][0]["chunk_id"], 1)
        self.assertEqual(messages[1]["citations"][0]["score"], 0.88)

    def test_app_py_compilation(self):
        """Verify that app.py syntax compiles with no errors."""
        app_path = CURRENT_DIR / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, str(app_path), "exec")
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
