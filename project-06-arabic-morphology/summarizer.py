"""Extractive Text Summarization Module.

This module provides a simple, transparent extractive text summarization algorithm
suitable for educational exploration in Arabic and English NLP.

Conceptual Pipeline:
1. Split input text into individual sentences.
2. Tokenize and compute word frequency distribution.
3. Score each sentence based on the significance of its constituent words.
4. Select the highest-ranking sentences.
5. Restore the original chronological sentence order to produce a coherent summary.
6. Compute word statistics and compression ratio.
"""

import re
from collections import Counter
from typing import Tuple

# Common high-frequency function words (stopwords) that do not carry core semantic content
STOPWORDS = {
    # English stopwords
    "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "of",
    "for", "with", "as", "by", "that", "this", "it", "are", "was", "were",
    "be", "been", "has", "have", "had", "or", "from", "they", "we", "you",
    # Arabic stopwords
    "في", "من", "على", "إلى", "أن", "عن", "مع", "هو", "هي", "هذا",
    "هذه", "كان", "كانت", "التي", "الذي", "الذين", "كل", "ما", "لا", "قد",
    "ثم", "أو", "إن", "بين", "حيث", "كما", "ذلك", "تلك", "هناك", "فإن"
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using English and Arabic sentence terminators."""
    # Split on newlines, periods, exclamation marks, question marks (English ? and Arabic ؟)
    raw_sentences = re.split(r"[\r\n]+|[.!?؟]+", text)
    return [s.strip() for s in raw_sentences if s.strip()]


def calculate_word_frequencies(sentences: list[str]) -> Counter:
    """Compute normalized term frequencies for content words across all sentences."""
    words = []
    for s in sentences:
        tokens = [
            w for w in re.findall(r"\w+", s.lower())
            if len(w) > 1 and w not in STOPWORDS
        ]
        words.extend(tokens)

    word_freq = Counter(words)

    # Fallback to all tokens if all words were filtered out as stopwords
    if not word_freq:
        all_tokens = [
            w for s in sentences
            for w in re.findall(r"\w+", s.lower())
            if len(w) > 1
        ]
        word_freq = Counter(all_tokens)

    return word_freq


def score_sentences(sentences: list[str], word_freq: Counter) -> list[tuple[float, int, str]]:
    """Score sentences based on the accumulated importance of their words."""
    max_freq = max(word_freq.values()) if word_freq else 1
    scored = []

    for idx, sentence in enumerate(sentences):
        tokens = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 1]
        if not tokens:
            scored.append((0.0, idx, sentence))
            continue

        # Sum normalized word frequencies
        raw_score = sum(word_freq.get(w, 0) / max_freq for w in tokens)
        # Normalize by length (square root) to avoid disproportionately favoring overly long sentences
        length_penalty = len(tokens) ** 0.5
        normalized_score = raw_score / length_penalty
        scored.append((normalized_score, idx, sentence))

    return scored


def summarize_text(text: str, ratio: float = 0.5) -> Tuple[int, str, int, str]:
    """Generate an extractive summary of the input text and compute metrics.

    Args:
        text: The raw input text (Arabic or English).
        ratio: Target proportion of sentences to retain (default 0.5 = 50%).

    Returns:
        Tuple containing:
        - original_word_count (int)
        - summary_text (str)
        - summary_word_count (int)
        - compression_ratio (str) e.g. "60.0%"
    """
    raw_text = text.strip() if text else ""

    # Edge case: Empty input
    if not raw_text:
        return 0, "", 0, "0.0%"

    orig_words = raw_text.split()
    orig_count = len(orig_words)

    # Edge case: Zero words
    if orig_count == 0:
        return 0, "", 0, "0.0%"

    sentences = split_sentences(raw_text)

    # Edge case: Very short text (1 sentence or cannot be meaningfully shortened)
    if len(sentences) <= 1:
        return orig_count, raw_text, orig_count, "0.0%"

    # 1. Compute word frequencies
    word_freq = calculate_word_frequencies(sentences)

    # 2. Score sentences
    scored_sentences = score_sentences(sentences, word_freq)

    # 3. Determine number of sentences to extract (at least 1, at most N-1)
    target_k = max(1, min(len(sentences) - 1, int(round(len(sentences) * ratio))))

    # 4. Select top scoring sentences
    top_k = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:target_k]

    # 5. Restore original sentence order
    selected = sorted(top_k, key=lambda x: x[1])

    # Reconstruct summary text
    summary = ".\n".join(s for _, _, s in selected) + "."
    summary_words = summary.split()
    summary_count = len(summary_words)

    # 6. Calculate compression ratio safely:
    # Compression Ratio = ((Original Word Count - Summary Word Count) / Original Word Count) * 100
    if orig_count > 0:
        reduction = max(0, orig_count - summary_count)
        compression_val = (reduction / orig_count) * 100
        compression_ratio = f"{compression_val:.1f}%"
    else:
        compression_ratio = "0.0%"

    return orig_count, summary, summary_count, compression_ratio
