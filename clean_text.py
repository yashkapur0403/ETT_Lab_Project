"""Text cleaning utilities for ESG report preprocessing."""

from __future__ import annotations

import re
from collections import Counter
from typing import List


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_standalone_page_numbers(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if re.fullmatch(r"page\s+\d{1,4}", stripped, flags=re.IGNORECASE):
            continue
        kept.append(line)
    return "\n".join(kept)


def _remove_repeated_lines(text: str, min_repeats: int = 5) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    counts = Counter(lines)
    repeated = {ln for ln, cnt in counts.items() if cnt >= min_repeats and len(ln) < 80}

    filtered: List[str] = []
    for raw_line in text.splitlines():
        if raw_line.strip() in repeated:
            continue
        filtered.append(raw_line)
    return "\n".join(filtered)


def clean_report_text(text: str) -> str:
    """Apply smart cleaning for maximum ESG extraction accuracy from tables.
    
    Optimized for speed while preserving table structure.
    """

    if not text:
        return ""

    # Step 1: Remove page numbers
    cleaned = _remove_standalone_page_numbers(text)
    
    # Step 2: Remove repeated lines (footers, headers)
    cleaned = _remove_repeated_lines(cleaned)
    
    # Step 3: Normalize whitespace
    cleaned = _normalize_whitespace(cleaned)
    
    return cleaned
    cleaned = _normalize_whitespace(cleaned)
    return cleaned
