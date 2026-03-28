"""Improved LLM-based ESG extraction with semantic context.

Enhances the basic LLM extraction with:
- Semantic search for relevant document chunks
- Context-aware prompts using retrieved chunks
- Better handling of tables
- Improved accuracy with semantic mapping
- Fallback to regex extraction for validation
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import InferenceClient

from app.pipeline.semantic_chunking import SemanticChunker, merge_chunks_for_context
from app.pipeline.semantic_search import SemanticSearchEngine, build_semantic_context
from app.pipeline.regex_validator import cross_validate_extraction


# Configuration
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
MAX_INPUT_CHARS = 4000
MAX_RETRIES = 1


def build_improved_prompt(
    clean_text: str,
    company_name: str,
    report_year: int,
    semantic_context: Optional[str] = None,
) -> str:
    """Build enhanced ESG extraction prompt with semantic context.

    Args:
        clean_text: Pre-cleaned report text
        company_name: Company name
        report_year: Report year
        semantic_context: Additional context from semantic search

    Returns:
        Formatted prompt for LLM
    """

    context_section = ""
    if semantic_context:
        context_section = f"""
SEMANTIC CONTEXT (Retrieved from document):
{semantic_context}

Use the semantic context above when available. It contains the most relevant
sections of the document for extracting ESG metrics.
"""

    prompt = f"""You are an expert ESG data extraction system specializing in parsing
sustainability and annual reports.

DOCUMENT METADATA:
- Company: {company_name}
- Report Year: {report_year}

{context_section}

EXTRACTION TASK:

Extract ESG metrics from the document text using these strategies in order:

1. ENERGY CONSUMPTION
   - Look for: MWh, kWh, GJ, TJ, BTU consumption values
   - Search tables with "Energy" or "Power" in headers
   - Find renewable vs total energy breakdown
   - Include units in extracted value (e.g., "15,000 MWh")
   - Pattern: [NUMBER] [UNIT]

2. WATER USAGE
   - Look for: ML, m³, liters, gallons of water consumed/withdrawn
   - Search "Water Management" or "Environmental" tables
   - Find recycled water percentage separately if available
   - Include units (e.g., "500 ML" or "25 million liters")
   - Pattern: [NUMBER] [UNIT]

3. TOTAL EMPLOYEES
   - Find headcount/FTE from HR or workforce tables
   - Look for "As of [DATE]" statements with employee counts
   - Can be in thousands (45.2K = 45,200)
   - Use latest reported period if multiple dates
   - Return clean number without commas
   - Pattern: [NUMBER] or [NUMBER]K

4. FEMALE EMPLOYEE PERCENTAGE (Diversity)
   - Find "Women in workforce" or "Female employees %" directly
   - If not found directly: Calculate Female Count ÷ Total = %
   - Example: "3,845 female out of 10,250 total" → 37.5%
   - Look in HR tables with gender breakdown
   - Return as percentage (e.g., "42.5%")
   - Pattern: [NUMBER]% or [FEMALE_COUNT] / [TOTAL_COUNT]

5. FEMALE BOARD MEMBER PERCENTAGE (Board Diversity)
   - Find "Women Directors %" or board composition table
   - Calculate Female Directors ÷ Total Directors = %
   - Example: "5 women out of 11 board members" → 45.5%
   - Look in "Board Composition" or "Governance" sections
   - Return as percentage (e.g., "45.5%")
   - Pattern: [NUMBER]% or [FEMALE_DIRECTORS] / [TOTAL_DIRECTORS]

CRITICAL RULES:

- DO read and interpret tables carefully
- DO calculate percentages from counts when needed
- DO include units with energy/water values
- DO preserve exact numbers - no rounding or estimates
- DO use latest reported values if multiple periods exist
- DO mark "Not Reported" ONLY if genuinely absent after thorough search
- DO NOT hallucinate - if uncertain, mark "Not Reported"
- DO NOT modify numbers - preserve as written in document
- DO NOT ignore table data

SEMANTIC UNDERSTANDING:

Beyond keywords:
- "We employ over 75,000 people" → 75,000 employees
- "Board: 3F, 8M" → 27.3% female (3 ÷ 11)
- "Renewable makes up 45% of our 2000 MWh" → 900 MWh renewable, 2000 MWh total
- "Gender split: 40F/60M" → 40% female
- Context phrases like "women comprise" or "female representation" indicate gender data

DOCUMENT TEXT:
---
{clean_text}
---

Return ONLY valid JSON (no markdown, no explanations):
{{
  "company": "{company_name}",
  "year": {report_year},
  "environment": {{
    "energy": "value with units or 'Not Reported'",
    "water": "value with units or 'Not Reported'"
  }},
  "social": {{
    "employees": "number or 'Not Reported'",
    "diversity": "percentage or 'Not Reported'"
  }},
  "governance": {{
    "board": "percentage or 'Not Reported'"
  }},
  "summary": "brief summary of what was found",
  "flags": []
}}

RESPOND WITH VALID JSON ONLY - NO ADDITIONAL TEXT.
"""
    return prompt


def extract_esg_with_semantic_search(
    document_text: str,
    pages: Optional[list] = None,
    tables: Optional[list] = None,
    company_name: str = "Unknown",
    report_year: int = 2024,
) -> Dict[str, Any]:
    """Extract ESG data using semantic search for context.

    Args:
        document_text: Full document text
        pages: Page data from extract_pdf_text_enhanced
        tables: Table data from extract_pdf_text_enhanced
        company_name: Company name
        report_year: Report year

    Returns:
        Extracted ESG data as dictionary
    """

    try:
        # Step 1: Chunk the document
        chunker = SemanticChunker(chunk_size=800, chunk_overlap=200)

        chunks = []
        if pages:
            chunks.extend(chunker.chunk_pages(pages))
        if tables:
            chunks.extend(chunker.chunk_tables(tables))
        if not chunks and document_text:
            chunks = chunker.chunk_text(document_text)

        if not chunks:
            chunks = [
                {
                    "id": 0,
                    "text": document_text,
                    "tokens": len(document_text) // 4,
                }
            ]

        # Step 2: Semantic search for ESG topics
        search_engine = SemanticSearchEngine()
        search_results = search_engine.search_esg_topics(chunks, top_k=3)

        # Step 3: Build semantic context for LLM
        semantic_context = build_semantic_context(search_results, max_context_chars=2000)

        # Step 4: Truncate document for LLM input
        truncated_text = document_text[:MAX_INPUT_CHARS]

        # Step 5: Build improved prompt
        prompt = build_improved_prompt(
            truncated_text,
            company_name,
            report_year,
            semantic_context,
        )

        # Step 6: Call LLM
        result = _call_llm(prompt)

        # Step 7: Validate and enhance with regex fallback
        if result and isinstance(result, dict):
            result = cross_validate_extraction(result, document_text, tables)

        return result or {}

    except Exception as e:
        return {
            "error": str(e),
            "company": company_name,
            "year": report_year,
        }


def _call_llm(prompt: str, max_retries: int = MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """Call HuggingFace LLM and parse JSON response.

    Args:
        prompt: Prompt for LLM
        max_retries: Number of retry attempts

    Returns:
        Parsed JSON response or None
    """

    try:
        api_key = os.environ.get("HF_API_KEY")
        if not api_key:
            return None

        client = InferenceClient(api_key=api_key, model=HF_MODEL)

        for attempt in range(max_retries):
            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=1000,
                    temperature=0.1,
                )

                # Parse JSON from response
                if isinstance(response, str):
                    response_text = response
                else:
                    response_text = response.generated_text if hasattr(response, 'generated_text') else str(response)

                # Extract JSON
                json_str = _extract_json_from_response(response_text)
                if json_str:
                    return json.loads(json_str)

            except Exception:
                if attempt == max_retries - 1:
                    raise

        return None

    except Exception as e:
        print(f"LLM error: {e}")
        return None


def _extract_json_from_response(response: str) -> Optional[str]:
    """Extract JSON object from LLM response text.

    Args:
        response: Raw response from LLM

    Returns:
        JSON string or None
    """

    # Find first { and last }
    start_idx = response.find("{")
    end_idx = response.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return response[start_idx : end_idx + 1]

    return None


def extract_esg_cached(
    document_text: str,
    pages: Optional[list] = None,
    tables: Optional[list] = None,
    company_name: str = "Unknown",
    report_year: int = 2024,
    cache_dir: str = "data/output",
) -> Dict[str, Any]:
    """Extract ESG with caching support.

    Args:
        document_text: Full document text
        pages: Page data
        tables: Table data
        company_name: Company name
        report_year: Report year
        cache_dir: Cache directory

    Returns:
        Extracted ESG data
    """

    # Generate cache key
    cache_key = hashlib.sha256(
        f"{document_text[:1000]}|{company_name}|{report_year}".encode()
    ).hexdigest()

    cache_path = Path(cache_dir) / f"esg_extraction_{cache_key}.json"

    # Check cache
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    # Extract
    result = extract_esg_with_semantic_search(
        document_text,
        pages,
        tables,
        company_name,
        report_year,
    )

    # Cache result
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    return result
