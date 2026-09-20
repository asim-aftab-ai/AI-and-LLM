"""Text Cleaner and Normalizer Module.

Cleans and standardizes extracted PDF text by removing whitespace noise,
fixing hyphenated line breaks, and preserving paragraph structure.
"""

import re
from typing import List, Tuple


def clean_text(raw_text: str) -> str:
    """Clean and normalize extracted textual content.

    Operations performed:
    1. Unify newline characters (\r\n and \r to \n).
    2. Join hyphenated words split across lines (e.g., 'arti-\nficial' -> 'artificial').
    3. Replace multiple horizontal spaces/tabs with a single space.
    4. Collapse 3+ consecutive line breaks into 2 line breaks to preserve clean paragraphs.
    5. Strip leading and trailing whitespace.

    Args:
        raw_text: Raw string extracted from document.

    Returns:
        Cleaned, normalized string.
    """
    if not raw_text:
        return ""

    # 1. Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Fix hyphenated word breaks at line ends (allowing optional whitespace before/after newline)
    text = re.sub(r"(\b[a-zA-Z]+)-[ \t]*\n[ \t]*([a-zA-Z]+\b)", r"\1\2", text)

    # 3. Replace non-breaking spaces and tabs with standard space
    text = text.replace("\u00a0", " ").replace("\t", " ")

    # 4. Collapse multiple horizontal whitespace on the same line
    text = re.sub(r"[ ]{2,}", " ", text)

    # 5. Remove trailing spaces on lines
    lines = [line.strip() for line in text.split("\n")]

    # 6. Collapse multiple consecutive empty lines (keep at most one blank line between paragraphs)
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if line == "":
            if not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    return "\n".join(cleaned_lines).strip()


def clean_pages_text(pages_text: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Clean text individually for each page while preserving page numbers."""
    cleaned_pages = []
    for page_num, raw_text in pages_text:
        cleaned = clean_text(raw_text)
        if cleaned:
            cleaned_pages.append((page_num, cleaned))
    return cleaned_pages
