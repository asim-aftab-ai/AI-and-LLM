"""Document Processor Module.

Handles document chunking for retrieval, metadata tracking,
and saving extracted text to disk.
"""

import os
from typing import List, Dict, Any, Tuple


def chunk_document(
    pages_text: List[Tuple[int, str]],
    chunk_size_words: int = 350,
    overlap_words: int = 50,
) -> List[Dict[str, Any]]:
    """Split cleaned document pages into overlapping semantic text chunks.

    Args:
        pages_text: List of (page_num, clean_text) tuples.
        chunk_size_words: Target maximum words per chunk.
        overlap_words: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk dictionaries with metadata (chunk_id, page_num, text, word_count).
    """
    chunks: List[Dict[str, Any]] = []
    chunk_id = 1

    for page_num, page_content in pages_text:
        words = page_content.split()
        if not words:
            continue

        # If page content fits in a single chunk, add it directly
        if len(words) <= chunk_size_words:
            chunks.append({
                "chunk_id": chunk_id,
                "page_num": page_num,
                "text": page_content.strip(),
                "word_count": len(words),
            })
            chunk_id += 1
            continue

        # Otherwise, slide a window across the page's words
        step = max(1, chunk_size_words - overlap_words)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size_words]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_num": page_num,
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                })
                chunk_id += 1

            # Stop if this was the last window of words
            if i + chunk_size_words >= len(words):
                break

    return chunks


def save_extracted_txt(cleaned_text: str, original_filename: str, output_dir: str = "data/extracted") -> str:
    """Save cleaned extracted text as a .txt file on disk.

    Args:
        cleaned_text: The normalized document text.
        original_filename: The original PDF filename.
        output_dir: Directory where the .txt file will be saved.

    Returns:
        str: Absolute or relative filepath of the saved .txt document.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    txt_filename = f"{base_name}_extracted.txt"
    filepath = os.path.join(output_dir, txt_filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    return filepath
