#!/usr/bin/env python3
"""CLI Script 01: Extract articles from raw law text."""

import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.extractor import extract_articles_to_csv


def main():
    parser = argparse.ArgumentParser(description="Extract legal articles from raw text file into structured CSV.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to raw law text file (e.g. law_text.txt)")
    parser.add_argument("--output", "-o", type=str, default="data/corpus/law_articles.csv", help="Output CSV path")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting articles from {args.input}...")
    num_articles = extract_articles_to_csv(args.input, args.output)
    print(f"Extracted {num_articles} articles into: {args.output}")


if __name__ == "__main__":
    main()
