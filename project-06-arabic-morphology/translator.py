"""Translation Utility for Bilingual Arabic-English Learning.

This module provides reliable Arabic-to-English translation capabilities designed
to serve as an educational learning aid alongside Arabic NLP processes.
"""

from functools import lru_cache
from deep_translator import MyMemoryTranslator, GoogleTranslator


def is_arabic_text(text: str) -> bool:
    """Check if the given string contains Arabic characters."""
    return any("\u0600" <= c <= "\u06FF" for c in text) if text else False


@lru_cache(maxsize=256)
def translate_to_english(text: str) -> str:
    """Translate Arabic text into English using reliable translation providers with fallbacks.

    If the text is already in English or Latin script, it is returned as-is.
    Results are cached in memory to optimize performance and prevent redundant requests.

    Args:
        text: Input text (Arabic or English).

    Returns:
        str: English translation or original English text.
    """
    cleaned = text.strip() if text else ""
    if not cleaned:
        return ""

    # If the input text contains no Arabic characters, it is already English/Latin
    if not is_arabic_text(cleaned):
        return cleaned

    # Provider 1: MyMemoryTranslator
    try:
        translated = MyMemoryTranslator(source="ar-SA", target="en-US").translate(cleaned)
        if translated and translated.strip():
            return translated.strip()
    except Exception:
        pass

    # Provider 2: GoogleTranslator fallback
    try:
        translated = GoogleTranslator(source="ar", target="en").translate(cleaned)
        if translated and translated.strip():
            return translated.strip()
    except Exception:
        pass

    # Offline / network fallback
    return "[Translation unavailable offline]"
