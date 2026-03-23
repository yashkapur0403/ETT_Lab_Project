"""
Simple, fast ESG extraction pipeline optimized for reliability.

Focuses on straightforward extraction without complex ranking:
- Extract text + tables
- Find candidates using regex
- Pick best by confidence (>0.5)
- Validate against hallucinations
"""

from __future__ import annotations

import time
from typing import Any, Dict

from app.pipeline.extract_text_enhanced import extract_pdf_text_enhanced
from app.pipeline.regex_validator import ESGRegexExtractor
from app.pipeline.value_validator import ESGValueValidator
from app.pipeline.smart_selector import SmartValueSelector
from app.models.canonical_schema import normalize_to_schema


def run_simple_pipeline(
    pdf_path: str,
    company_name: str = "Unknown",
    report_year: int = 2024,
) -> Dict[str, Any]:
    """Simple, reliable ESG extraction pipeline.

    Args:
        pdf_path: Path to PDF file
        company_name: Company name
        report_year: Report year

    Returns:
        Extracted ESG data matching canonical schema
    """
    t0 = time.time()

    print(f"[1/4] Extracting text and tables from PDF...")
    t1 = time.time()
    extraction = extract_pdf_text_enhanced(pdf_path)
    dt1 = time.time() - t1
    print(f"  └─ Took {dt1:.1f}s")

    if extraction.get("errors"):
        print(f"  Warnings: {extraction['errors']}")

    document_text = extraction.get("full_text", "")
    tables = extraction.get("tables", [])

    print(f"[2/4] Finding ESG metrics...")
    t2 = time.time()

    # Extract from text
    extractor = ESGRegexExtractor()
    metrics_text = extractor.extract_all_metrics(document_text)

    # Extract from tables
    metrics_tables = {
        "energy": [],
        "water": [],
        "employees": [],
        "diversity": [],
        "board": [],
    }

    for table in tables:
        table_text = _table_to_text(table)
        table_metrics = extractor.extract_all_metrics(table_text)
        for field in metrics_tables:
            metrics_tables[field].extend(table_metrics.get(field, []))

    dt2 = time.time() - t2
    print(f"  └─ Took {dt2:.1f}s")

    print(f"[3/4] Selecting best candidates with smart filtering...")
    t3 = time.time()

    # Combine all candidates from text and tables
    all_candidates = {
        "energy": metrics_text.get("energy", []) + metrics_tables.get("energy", []),
        "water": metrics_text.get("water", []) + metrics_tables.get("water", []),
        "employees": metrics_text.get("employees", []) + metrics_tables.get("employees", []),
        "diversity": metrics_text.get("diversity", []) + metrics_tables.get("diversity", []),
        "board": metrics_text.get("board", []) + metrics_tables.get("board", []),
    }

    # Use smart selector to filter out partials and pick best
    selector = SmartValueSelector()
    esg_dict = selector.select_all_fields(all_candidates, document_text)

    # Fill "Not Reported" for None values
    for field in ["energy", "water", "employees", "diversity", "board"]:
        if esg_dict[field] is None:
            esg_dict[field] = "Not Reported"

    dt3_before = time.time() - t3

    # Validate values to catch hallucinations as final safety check
    validator = ESGValueValidator()
    esg_dict = validator.validate_all(esg_dict, document_text)

    dt3 = time.time() - t3
    print(f"  └─ Took {dt3:.1f}s (validation: {dt3 - dt3_before:.1f}s)")

    print(f"[4/4] Preparing output...")

    # Build ESG data
    esg_data = {
        "company": company_name,
        "year": report_year,
        "environment": {
            "energy": esg_dict.get("energy", "Not Reported"),
            "water": esg_dict.get("water", "Not Reported"),
        },
        "social": {
            "employees": esg_dict.get("employees", "Not Reported"),
            "diversity": esg_dict.get("diversity", "Not Reported"),
        },
        "governance": {
            "board": esg_dict.get("board", "Not Reported"),
        },
        "summary": f"Extracted from {extraction.get('page_count', 0)} pages with {len(tables)} tables.",
        "flags": _build_flags(esg_dict, extraction),
    }

    # Normalize to schema
    normalized = normalize_to_schema(esg_data)

    # Add metadata
    normalized["extraction_metadata"] = {
        "pdf_path": extraction.get("pdf_path"),
        "pages": extraction.get("page_count"),
        "extraction_quality": extraction.get("extraction_quality"),
        "tables_found": len(tables),
        "method": "simple_regex_based",
    }

    total_time = time.time() - t0
    print(f"  └─ Total time: {total_time:.1f}s")

    return normalized


def _table_to_text(table: Dict) -> str:
    """Convert table dict to text for regex extraction."""
    lines = []
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    for row in rows:
        for header in headers:
            value = row.get(header, "")
            if value:
                lines.append(f"{header} {value}")

    return " ".join(lines)


def _build_flags(metrics: Dict, extraction: Dict) -> list:
    """Build quality flags."""
    flags = []

    quality = extraction.get("extraction_quality", "poor")
    if quality in ["poor", "fair"]:
        flags.append({
            "issue": f"PDF extraction quality is '{quality}' - results may be incomplete",
            "severity": "warning"
        })

    not_reported = sum(
        1 for v in metrics.values() if v == "Not Reported"
    )
    if not_reported >= 3:
        flags.append({
            "issue": "Multiple metrics not found - document may not contain ESG data",
            "severity": "warning"
        })

    return flags


# Aliases for compatibility
run_fast_pipeline = run_simple_pipeline
run_improved_pipeline = run_simple_pipeline
