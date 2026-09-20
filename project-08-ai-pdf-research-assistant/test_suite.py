"""
Automated Test Suite for Project 08: AI PDF Research Assistant
Verifies all pipeline stages: PDF validation, extraction, cleaning,
chunking, TXT export, local semantic retrieval, and conversation memory.
"""

import io
import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processor import validate_and_extract_pdf
from src.text_cleaner import clean_text, clean_pages_text
from src.document_processor import chunk_document, save_extracted_txt
from src.retriever import DocumentRetriever, format_context_for_prompt
from src.conversation import ConversationManager
from src.llm import get_llm_client, SYSTEM_PROMPT


def create_sample_pdf_bytes(text_lines: list[str]) -> bytes:
    """Generates a valid raw PDF in memory with text lines."""
    stream_content = "BT\n/F1 12 Tf\n"
    y_pos = 700
    for line in text_lines:
        safe_line = line.replace("(", "\\(").replace(")", "\\)")
        stream_content += f"72 {y_pos} Td ({safe_line}) Tj\n0 -20 Td\n"
        y_pos -= 20
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


class TestPDFProcessor(unittest.TestCase):
    def test_invalid_pdf_header(self):
        invalid_bytes = b"This is not a PDF file at all."
        res = validate_and_extract_pdf(invalid_bytes, "invalid.txt")
        self.assertFalse(res["success"])
        self.assertIn("missing %PDF header", res["error"])

    def test_empty_bytes(self):
        res = validate_and_extract_pdf(b"", "empty.pdf")
        self.assertFalse(res["success"])
        self.assertIn("empty", res["error"])

    def test_valid_pdf_extraction(self):
        pdf_bytes = create_sample_pdf_bytes([
            "Transformer architectures rely entirely on self-attention mechanisms.",
            "They discard recurrence and convolutions to achieve superior parallelization."
        ])
        res = validate_and_extract_pdf(pdf_bytes, "test_paper.pdf")
        self.assertTrue(res["success"], res.get("error"))
        self.assertEqual(res["page_count"], 1)
        self.assertIn("Transformer architectures", res["full_text"])
        self.assertEqual(len(res["pages_text"]), 1)


class TestTextCleaner(unittest.TestCase):
    def test_cleaning_whitespace_and_hyphens(self):
        raw = "This is a demon-\nstration of text clean-   \ning and    redundant   spaces.\n\n\n\nNext paragraph."
        cleaned = clean_text(raw)
        self.assertIn("demonstration", cleaned)
        self.assertIn("cleaning", cleaned)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_clean_pages_text(self):
        pages = [(1, "Page 1 con-\ntent with    spaces.")]
        cleaned_pages = clean_pages_text(pages)
        self.assertEqual(len(cleaned_pages), 1)
        self.assertEqual(cleaned_pages[0][0], 1)
        self.assertEqual(cleaned_pages[0][1], "Page 1 content with spaces.")


class TestDocumentProcessor(unittest.TestCase):
    def test_save_extracted_txt(self):
        sample_text = "Standard extracted content for testing text export capabilities."
        saved_path = save_extracted_txt(sample_text, "export_sample.pdf", output_dir="data/extracted")
        self.assertTrue(os.path.exists(saved_path))
        with open(saved_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, sample_text)

    def test_chunking_with_overlap(self):
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        pages_text = [(1, text)]
        chunks = chunk_document(pages_text, chunk_size_words=40, overlap_words=10)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        # Verify overlap between chunk 1 and chunk 2
        chunk1_words = chunks[0]["text"].split()
        chunk2_words = chunks[1]["text"].split()
        self.assertEqual(chunk1_words[-10:], chunk2_words[:10])


class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = DocumentRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_indexing_and_similarity_retrieval(self):
        chunks = [
            {"chunk_id": 1, "text": "The solar system contains eight planets orbiting the Sun.", "page_num": 1},
            {"chunk_id": 2, "text": "Deep neural networks learn hierarchical representations from large datasets.", "page_num": 2},
            {"chunk_id": 3, "text": "Mars is the fourth planet from the Sun and has a thin atmosphere.", "page_num": 3},
        ]
        self.retriever.index_chunks(chunks)
        results = self.retriever.retrieve("Tell me about planets and the Sun", top_k=2)
        self.assertEqual(len(results), 2)
        # Planetary chunks should have higher similarity than deep neural networks
        ids = [r["chunk_id"] for r in results]
        self.assertIn(1, ids)
        self.assertIn(3, ids)
        self.assertNotIn(2, ids)
        self.assertGreater(results[0]["score"], 0.3)

    def test_format_context_for_prompt(self):
        chunks = [
            {"chunk_id": 1, "page_num": 1, "score": 0.85, "text": "Self-attention mechanism excerpt."}
        ]
        context_block = format_context_for_prompt(chunks)
        self.assertIn("Page 1", context_block)
        self.assertIn("Chunk 1", context_block)
        self.assertIn("Self-attention mechanism excerpt.", context_block)


class TestConversationManager(unittest.TestCase):
    def test_conversation_turns(self):
        conv = ConversationManager()
        self.assertFalse(conv.has_messages())

        conv.add_user_message("What is this document about?")
        conv.add_assistant_message(
            "It is about solar system astronomy.",
            citations=[{"chunk_id": 1, "page_num": 1, "score": 0.85, "text": "solar system..."}]
        )
        self.assertTrue(conv.has_messages())
        all_msgs = conv.get_all_messages()
        self.assertEqual(len(all_msgs), 2)
        self.assertEqual(all_msgs[0]["content"], "What is this document about?")

        # Add a follow-up user query
        conv.add_user_message("Why is Mars red?")
        # get_messages_for_llm should return the preceding turns
        llm_msgs = conv.get_messages_for_llm(max_turns=3)
        self.assertEqual(len(llm_msgs), 2)
        self.assertEqual(llm_msgs[0]["role"], "user")
        self.assertEqual(llm_msgs[1]["role"], "assistant")

        conv.clear()
        self.assertFalse(conv.has_messages())


class TestLLMConfiguration(unittest.TestCase):
    def test_system_prompt_grounding(self):
        self.assertIn("CRITICAL INSTRUCTIONS", SYSTEM_PROMPT)
        self.assertIn("Ground your answers ONLY in the provided document context", SYSTEM_PROMPT)
        self.assertIn("this information was not found", SYSTEM_PROMPT)

    def test_client_init(self):
        client = get_llm_client(api_key="test-key-placeholder")
        self.assertEqual(str(client.base_url), "https://openrouter.ai/api/v1/")


if __name__ == "__main__":
    unittest.main(verbosity=2)
