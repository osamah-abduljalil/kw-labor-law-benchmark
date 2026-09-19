"""Text normalization, tokenization, and chunking for Arabic legal corpora."""

import re
from typing import List


def normalize_arabic(text: str) -> str:
    """Light normalization for Arabic retrieval while preserving legal phrasing.

    Args:
        text: Raw Arabic string.

    Returns:
        Cleaned, whitespace-normalized string.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Normalize excessive whitespaces and tabs/newlines
    text = re.sub(r"\s+", " ", text)
    return text


def clean_question(question: str) -> str:
    """Removes common dataset question prefixes such as 'س 1 /' or 'س1/'."""
    if not isinstance(question, str):
        return ""
    cleaned = re.sub(r"س\s*\d+\s*/\s*", "", question)
    return cleaned.strip()


def simple_tokenize(text: str) -> List[str]:
    """Basic whitespace tokenization for BM25 sparse retrieval.

    Args:
        text: Input string.

    Returns:
        List of tokens.
    """
    return normalize_arabic(text).split()


def chunk_text_words(
    text: str,
    chunk_words: int = 220,
    overlap_words: int = 40
) -> List[str]:
    """Splits text into overlapping word chunks.

    Args:
        text: Input text string.
        chunk_words: Maximum words per chunk.
        overlap_words: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = normalize_arabic(text).split()
    if len(words) <= chunk_words:
        return [normalize_arabic(text)]

    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_words]
        chunks.append(" ".join(chunk))
        if i + chunk_words >= len(words):
            break
        i += max(1, chunk_words - overlap_words)
    return chunks
