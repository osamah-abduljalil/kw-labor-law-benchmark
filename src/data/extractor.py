"""Extractor to parse raw legal text into structured article records."""

import csv
import re
from pathlib import Path
from typing import Dict, List, Union


def extract_articles_from_text(raw_text: str) -> List[Dict[str, str]]:
    """Extracts statutory articles from raw text using regex pattern.

    Args:
        raw_text: Full string content of raw law text.

    Returns:
        List of dicts containing 'article_number' and 'article_text'.
    """
    # Normalize line endings
    text = raw_text.replace("\r", "")
    # Normalize separators (---)
    text = re.sub(r"-{3,}", "\n", text)

    article_pattern = re.compile(
        r"مادة\s*(\d+)\s*(.*?)"
        r"(?=\nمادة\s*\d+|\Z)",
        re.DOTALL
    )

    records = []
    for match in article_pattern.finditer(text):
        article_number = match.group(1).strip()
        article_text = match.group(2).strip()

        # Normalize whitespace while keeping content intact
        article_text = re.sub(r"\n+", " ", article_text)
        article_text = re.sub(r"\s{2,}", " ", article_text)

        records.append({
            "article_number": article_number,
            "article_text": article_text
        })

    return records


def extract_articles_to_csv(
    input_file: Union[str, Path],
    output_file: Union[str, Path]
) -> int:
    """Parses raw law text file and saves extracted articles to a CSV file.

    Args:
        input_file: Path to raw law text file.
        output_file: Path where extracted articles CSV should be written.

    Returns:
        Number of articles extracted.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    records = extract_articles_from_text(content)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["article_number", "article_text"])
        writer.writeheader()
        writer.writerows(records)

    return len(records)
