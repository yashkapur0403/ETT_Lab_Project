"""Enhanced PDF text extraction with structure-aware table parsing.

This improved extractor uses pdfplumber for superior table extraction and
structure preservation compared to basic PyMuPDF text extraction.

Features:
- Extracte both raw text and structured tables
- Preserve table column/row relationships
- Handle multi-page documents
- Detect table headers and data rows correctly
- Return metadata about extraction quality
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def extract_pdf_text_enhanced(pdf_path: str) -> Dict[str, Any]:
    """Extract text with structure-aware table parsing using pdfplumber.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict containing:
        - pdf_path: Input file path
        - page_count: Total pages
        - full_text: Concatenated text from all pages
        - pages: List of page data with text and tables
        - tables: Flattened list of all tables with page references
        - extraction_quality: Assessment of extraction success
        - errors: List of any extraction errors
    """

    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required. Install with: pip install pdfplumber"
        ) from exc

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    all_tables = []
    full_text_parts = []
    errors = []

    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_data = _extract_page_data(page, page_num, all_tables)
                pages.append(page_data)

                # Collect text for full document
                if page_data["text"]:
                    full_text_parts.append(page_data["text"])

    except Exception as e:
        errors.append(f"PDF parsing error: {str(e)}")

    full_text = "\n\n".join(f for f in full_text_parts if f)

    # Assess extraction quality
    quality = _assess_extraction_quality(full_text, all_tables)

    return {
        "pdf_path": str(path),
        "page_count": len(pages),
        "full_text": full_text,
        "pages": pages,
        "tables": all_tables,
        "extraction_quality": quality,
        "errors": errors,
    }


def _extract_page_data(
    page: Any,
    page_num: int,
    all_tables: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract text and tables from a single page.

    Args:
        page: pdfplumber page object
        page_num: Page number (1-indexed)
        all_tables: List to collect tables across all pages

    Returns:
        Dictionary with page metadata and content
    """

    # Extract raw text
    text = page.extract_text() or ""
    text = text.strip()

    # Extract tables
    tables = []
    try:
        raw_tables = page.extract_tables()
        if raw_tables:
            for table_idx, table in enumerate(raw_tables):
                table_data = _convert_table_to_dict(table, page_num, table_idx)
                tables.append(table_data)
                all_tables.append(table_data)
    except Exception as e:
        # Log but don't fail - continue with text extraction
        pass

    return {
        "page_number": page_num,
        "text": text,
        "char_count": len(text),
        "table_count": len(tables),
        "tables": tables,
    }


def _convert_table_to_dict(
    table: List[List[Optional[str]]],
    page_num: int,
    table_idx: int,
) -> Dict[str, Any]:
    """Convert table from list of lists to structured dictionary.

    Preserves header row and key-value relationships for easier searching.

    Args:
        table: Table as list of lists (rows)
        page_num: Page number where table appears
        table_idx: Index of table on page

    Returns:
        Structured table dict with headers and rows
    """

    if not table or len(table) < 1:
        return {"page": page_num, "index": table_idx, "rows": []}

    # First row as header
    headers = table[0]
    rows = []

    for row_idx, row in enumerate(table[1:], start=1):
        row_dict = {}
        for col_idx, (header, value) in enumerate(zip(headers, row)):
            # Clean up values
            clean_header = (header or "").strip()
            clean_value = (value or "").strip()
            if clean_header:
                row_dict[clean_header] = clean_value

        if row_dict:  # Only add non-empty rows
            rows.append(row_dict)

    # Also create markdown representation for readability
    markdown_table = _table_to_markdown(headers, table[1:])

    return {
        "page": page_num,
        "index": table_idx,
        "headers": [h.strip() if h else "" for h in headers],
        "rows": rows,
        "markdown": markdown_table,
    }


def _table_to_markdown(headers: List[Optional[str]], rows: List[List[Optional[str]]]) -> str:
    """Convert table to markdown format for display/searching.

    Args:
        headers: Header row
        rows: Data rows

    Returns:
        Markdown-formatted table string
    """

    if not headers:
        return ""

    # Clean headers
    clean_headers = [str(h or "").strip() for h in headers]

    # Start markdown
    lines = ["| " + " | ".join(clean_headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(clean_headers)) + "|")

    # Add rows
    for row in rows:
        clean_row = [str(cell or "").strip() for cell in row]
        lines.append("| " + " | ".join(clean_row) + " |")

    return "\n".join(lines)


def _assess_extraction_quality(full_text: str, tables: List[Dict[str, Any]]) -> str:
    """Assess quality of PDF extraction.

    Args:
        full_text: Extracted text
        tables: Extracted tables

    Returns:
        Quality assessment: "excellent", "good", "fair", or "poor"
    """

    text_len = len(full_text)
    table_count = len(tables)

    if text_len < 500 and table_count == 0:
        return "poor"
    elif text_len < 2000 and table_count == 0:
        return "fair"
    elif text_len >= 5000 and table_count > 0:
        return "excellent"
    elif text_len >= 2000 and table_count > 0:
        return "good"
    else:
        return "fair"


def extract_table_as_text(table: Dict[str, Any]) -> str:
    """Convert extracted table dict back to readable text format.

    Useful for passing to LLM for semantic understanding.

    Args:
        table: Table dict from extract_pdf_text_enhanced

    Returns:
        Human-readable table text
    """

    lines = []

    # Add page info
    lines.append(f"Table from Page {table.get('page', 'unknown')}:")

    # Add markdown representation if available
    if "markdown" in table:
        lines.append(table["markdown"])
    elif "rows" in table:
        # Fallback: reconstruct from rows
        headers = table.get("headers", [])
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")

            for row in table["rows"]:
                row_values = [str(row.get(h, "")) for h in headers]
                lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines)


def extract_tables_as_text(tables: List[Dict[str, Any]]) -> str:
    """Convert all extracted tables to readable text.

    Args:
        tables: List of table dicts

    Returns:
        Concatenated text representation of all tables
    """

    return "\n\n---\n\n".join(extract_table_as_text(t) for t in tables)


def search_tables_by_keyword(
    tables: List[Dict[str, Any]],
    keywords: List[str],
    case_sensitive: bool = False,
) -> List[Tuple[Dict[str, Any], List[str]]]:
    """Search tables for keywords in headers or values.

    Args:
        tables: List of extracted tables
        keywords: Keywords to search for
        case_sensitive: Whether search is case-sensitive

    Returns:
        List of (table, matching_keywords) tuples
    """

    results = []

    for table in tables:
        matching = []

        # Search headers
        headers = table.get("headers", [])
        for keyword in keywords:
            for header in headers:
                if _keyword_match(keyword, header, case_sensitive):
                    matching.append(f"header: {header}")

        # Search values
        rows = table.get("rows", [])
        for row in rows:
            for value in row.values():
                for keyword in keywords:
                    if _keyword_match(keyword, value, case_sensitive):
                        matching.append(f"value: {value}")

        if matching:
            results.append((table, matching))

    return results


def _keyword_match(keyword: str, text: str, case_sensitive: bool) -> bool:
    """Check if keyword appears in text.

    Args:
        keyword: Search term
        text: Text to search
        case_sensitive: Case sensitivity

    Returns:
        True if keyword found
    """

    if case_sensitive:
        return keyword in text
    else:
        return keyword.lower() in text.lower()
