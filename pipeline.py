# pipeline.py - PDF processing, indexing, and LLM extraction

import io
import re
import json
import os
import requests
from dataclasses import dataclass, field
from typing import Optional

from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

def safe_llm_call(messages, max_tokens=800, llm_token=None):
    """
    Safe LLM call with automatic model fallback and error handling.
    
    Default settings optimized for metric extraction:
    - max_tokens: 800 (high enough for complete responses)
    - temperature: 0.1 (deterministic, focused answers)
    """
    api_key = llm_token or os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("❌ ERROR: Groq API key is missing")
        return None
        
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Groq client: {e}")
        return None
    
    # Try multiple models in order
    models_to_try = [
        "mixtral-8x7b-32768",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama2-70b-4096",
    ]
    
    for model_idx, model in enumerate(models_to_try):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1  # Low temperature for focused, deterministic responses
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            error_str = str(e)
            if "decommissioned" in error_str.lower() or "not found" in error_str.lower():
                continue
            elif "401" in error_str or "authentication" in error_str.lower():
                print(f"❌ ERROR: Invalid API key")
                return None
            elif "429" in error_str or "rate" in error_str.lower():
                print(f"❌ ERROR: Quota exhausted")
                return None
            else:
                if model_idx == len(models_to_try) - 1:
                    print(f"❌ ERROR: All LLM models failed")
                continue
    
    return None


@dataclass
class ESGMetrics:
    # Environmental
    ghg_scope1: Optional[float] = None          # tCO2e
    ghg_scope2: Optional[float] = None          # tCO2e
    energy_consumption: Optional[float] = None  # GJ
    renewable_energy_pct: Optional[float] = None # 0-100
    water_withdrawal: Optional[float] = None    # KL
    waste_generated: Optional[float] = None     # MT
    waste_recycled_pct: Optional[float] = None  # 0-100
    # Social
    total_employees: Optional[int] = None
    women_employees_pct: Optional[float] = None # 0-100
    training_hours_avg: Optional[float] = None
    lost_time_injury_rate: Optional[float] = None
    csr_spend: Optional[float] = None           # INR crore
    # Governance
    board_size: Optional[int] = None
    independent_directors_pct: Optional[float] = None # 0-100
    women_directors_pct: Optional[float] = None       # 0-100
    audit_committee_independent: Optional[bool] = None
    whistleblower_policy: Optional[bool] = None
    # Meta
    company_name: Optional[str] = None
    reporting_year: Optional[str] = None
    framework: Optional[str] = None
    raw_text_sample: str = ""


@dataclass
class ESGScore:
    environmental: float
    social: float
    governance: float
    overall: float
    grade: str


@dataclass
class Gap:
    category: str
    field: str
    severity: str
    message: str


@dataclass
class InsightReport:
    score: ESGScore
    gaps: list
    llm_insights: str
    highlights: list
    red_flags: list
    recommendations: list


def _detect_and_normalize_tables(raw_text: str) -> str:
    """
    Detect table-like structures and convert to readable lines.
    
    Converts:
    | Metric | Value |
    | GHG Scope 1 | 1234 |
    
    To:
    GHG Scope 1: 1234
    """
    lines = raw_text.split('\n')
    normalized = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect table separator lines (ignore them)
        if re.match(r'^[\s\-_|+=]+$', line):
            i += 1
            continue
        
        # Detect table rows (contain | as separator)
        if '|' in line:
            # Split by pipe and clean
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]  # Remove empty parts
            
            # If 2 parts, treat as "Key: Value"
            if len(parts) == 2 and any(c.isdigit() for c in parts[1]):
                normalized.append(f"{parts[0]}: {parts[1]}")
            # If more parts, try to merge intelligently
            elif len(parts) > 2:
                # Assume first part is key, rest is value
                key = parts[0]
                value = ' '.join(parts[1:])
                normalized.append(f"{key}: {value}")
            else:
                # Just join with spaces
                normalized.append(' '.join(parts))
        else:
            # Regular line
            if line:
                normalized.append(line)
        
        i += 1
    
    return '\n'.join(normalized)


def _merge_broken_lines(lines: list[str]) -> list[str]:
    """
    Merge broken lines where a logical line was split across multiple lines.
    
    Example:
    ["Total", "Employees", "= 75323"]
    → ["Total Employees = 75323"]
    """
    merged = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Check if this line is incomplete (ends with partial word, number, operator)
        # If next line starts with lowercase or is a number/operator, merge
        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line:
                i += 1
                continue
            
            # Conditions to merge:
            # 1. Current line ends with lowercase and next starts with lowercase
            # 2. Current line ends with number/special char and next is number/unit
            # 3. Current line is single word and next starts lowercase
            
            should_merge = False
            
            if line and next_line:
                last_char = line[-1]
                first_char = next_line[0]
                
                # If current line is a single short word or a number
                if len(line.split()) <= 2 or re.match(r'^\d+', line):
                    # Merge with next line if it contains text or numbers
                    if re.match(r'^[a-z\d]', next_line, re.IGNORECASE):
                        should_merge = True
                
                # If line ends with operator or number, and next starts with number/unit
                if re.match(r'[=:,\-]$', last_char) or last_char.isdigit():
                    if first_char.isdigit() or first_char == '%':
                        should_merge = True
            
            if should_merge:
                line = f"{line} {next_line}"
                i += 1
            else:
                break
        
        merged.append(line)
        i += 1
    
    return merged


def _tag_metric_line(text: str) -> str | None:
    """
    Tag a line with ESG category if it contains metric keywords.
    
    Returns tag string (ENV_GHG, ENV_WATER, etc.) or None if not tagged.
    """
    text_lower = text.lower()
    
    # Must contain number
    if not re.search(r'\d', text):
        return None
    
    # Check ESG keywords
    if any(kw in text_lower for kw in ['scope 1', 'ghg', 'emissions', 'co2', 'tco2']):
        return 'ENV_GHG'
    elif any(kw in text_lower for kw in ['water', 'withdrawal', 'consumption']):
        return 'ENV_WATER'
    elif any(kw in text_lower for kw in ['energy', 'electricity', 'kwh', 'gj', 'mwh']):
        return 'ENV_ENERGY'
    elif any(kw in text_lower for kw in ['employee', 'workforce', 'headcount', 'staff']):
        return 'SOCIAL_EMP'
    elif any(kw in text_lower for kw in ['women', 'female', 'diversity', 'gender']):
        return 'SOCIAL_DIV'
    elif any(kw in text_lower for kw in ['board', 'director', 'independent']):
        return 'GOV_BOARD'
    
    return None


def preprocess_pdf_text(raw_text: str, max_chars: int = 20000) -> tuple[str, list[dict]]:
    """
    PRODUCTION-GRADE preprocessing with:
    1. Noise removal (page numbers, headers, footers)
    2. Table normalization (convert tables to readable lines)
    3. Broken line merging (fix line breaks in the middle of text)
    4. Metric tagging (tag by ESG category)
    5. Relevance filtering (only keep lines with numbers + keywords)
    6. Structured output (clean text + tagged lines)
    
    Returns:
    - cleaned_text: str (for FAISS embedding)
    - tagged_lines: list[dict] (for metric extraction)
      Format: [{"tag": "ENV_GHG", "text": "Scope 1 emissions: 1234 tCO2e"}, ...]
    """
    print("\n📋 PRODUCTION-GRADE TEXT PREPROCESSING")
    print(f"   Input: {len(raw_text)} chars")
    
    # STEP 0: Table normalization (before line processing)
    print("   🔄 Step 0: Detecting and normalizing tables...")
    raw_text = _detect_and_normalize_tables(raw_text)
    
    lines = raw_text.split('\n')
    
    # STEP 1: Remove noise lines
    print("   🔄 Step 1: Removing noise (page numbers, headers, legal text)...")
    cleaned_lines = []
    page_number_pattern = re.compile(r'^\s*(?:Page\s+\d+|P\s+\d+|\|\s*\d+\s*\||\d+\s*\||\|\s*\d+)\s*$')
    header_footer_pattern = re.compile(r'^(BRSR|ESG|Report|Annual|Sustainability|Disclosure|Corporate|Governance)$', re.IGNORECASE)
    noise_patterns = [
        r'^https?://',  # URLs
        r'^\d+[-\.]?\d+[-\.]?\d+[-\.]?\d+',  # IP addresses
        r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}',  # Email
        r'^\+?\d{1,3}[\s.-]?\d{3,}',  # Phone
        r'©|®|™',  # Trademark symbols
    ]
    
    for line in lines:
        line = line.strip()
        
        if not line or len(line) < 3:
            continue
        
        # Skip page numbers
        if page_number_pattern.match(line):
            continue
        
        # Skip pure headers/footers
        if header_footer_pattern.match(line):
            continue
        
        # Skip noise patterns
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in noise_patterns):
            continue
        
        # Skip lines that are mostly symbols
        if len(re.sub(r'[^a-zA-Z0-9\s%-]', '', line)) < len(line) * 0.3:
            continue
        
        # Skip legal disclaimers
        if any(phrase in line.lower() for phrase in ['confidential', 'proprietary', 'disclaimer', 'copyright']):
            if len(re.findall(r'\d', line)) < 2:  # Unless it has metrics
                continue
        
        cleaned_lines.append(line)
    
    print(f"      ✓ {len(cleaned_lines)}/{len(lines)} lines after noise removal")
    
    # STEP 2: Merge broken lines
    print("   🔄 Step 2: Merging broken lines...")
    merged_lines = _merge_broken_lines(cleaned_lines)
    print(f"      ✓ Merged to {len(merged_lines)} logical lines")
    
    # STEP 3: Normalize and clean
    print("   🔄 Step 3: Normalizing text...")
    normalized = []
    for line in merged_lines:
        # Normalize whitespace
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Remove excessive special characters but keep colons, dashes, parentheses
        line = re.sub(r'[^\w\s\.\,\%\-\(\)/:=&;]', '', line)
        
        if line and len(line) > 2:
            normalized.append(line)
    
    print(f"      ✓ {len(normalized)} lines after normalization")
    
    # STEP 4: Tag metrics by ESG category
    print("   🔄 Step 4: Tagging metrics by ESG category...")
    tagged_lines = []
    for line in normalized:
        tag = _tag_metric_line(line)
        if tag:
            tagged_lines.append({
                "tag": tag,
                "text": line
            })
    
    print(f"      ✓ {len(tagged_lines)} metric lines tagged")
    if tagged_lines:
        print(f"         Sample tags: {', '.join(set(t['tag'] for t in tagged_lines[:5]))}")
        print(f"         Sample line: {tagged_lines[0]['text'][:100]}")
    
    # STEP 5: Filter for relevance (only lines with numbers + keywords)
    print("   🔄 Step 5: Filtering for relevance...")
    relevant_lines = []
    esg_keywords = [
        'energy', 'electricity', 'fuel', 'renewable', 'water', 'withdrawal',
        'emissions', 'ghg', 'scope', 'co2', 'carbon', 'employee', 'workforce',
        'women', 'diversity', 'gender', 'board', 'director', 'governance',
        'waste', 'recycl', 'safety', 'health', 'injury', 'audit', 'policy',
    ]
    
    for line in normalized:
        line_lower = line.lower()
        has_number = bool(re.search(r'\d', line))
        has_keyword = any(kw in line_lower for kw in esg_keywords)
        has_metrics = bool(re.search(r'\d+\s*(?:%|tonnes|mt|kwh|gj|litre|people|employee|employees)', line_lower))
        
        if has_number and (has_keyword or has_metrics):
            relevant_lines.append(line)
    
    print(f"      ✓ {len(relevant_lines)} lines after relevance filter")
    
    # STEP 6: Build cleaned text for embedding
    print("   🔄 Step 6: Building cleaned text...")
    cleaned_text = ' '.join(relevant_lines)
    
    # Remove excessive repetition
    cleaned_text = re.sub(r'(\w+)\s+\1{2,}', r'\1', cleaned_text, flags=re.IGNORECASE)
    
    # HARD LIMIT: Truncate to max_chars with priority to metrics
    if len(cleaned_text) > max_chars:
        print(f"      ⚠️  Exceeds {max_chars}, truncating with priority...")
        sentences = [s.strip() for s in re.split(r'[.!?]', cleaned_text) if s.strip()]
        
        def score_sentence(s):
            score = 0
            if re.search(r'\d+\s*(?:tonnes|mt|kwh|gj|litre|%)', s, re.IGNORECASE):
                score += 100
            if any(kw in s.lower() for kw in ['ghg', 'scope', 'emissions', 'women', 'board']):
                score += 25
            return score
        
        sentences.sort(key=score_sentence, reverse=True)
        
        cleaned_text_truncated = ""
        for sent in sentences:
            if len(cleaned_text_truncated) + len(sent) + 2 <= max_chars:
                cleaned_text_truncated += sent + ". "
        
        cleaned_text = cleaned_text_truncated.strip()
    
    print(f"      ✓ Final text: {len(cleaned_text)} chars")
    
    # STEP 7: Debug summary
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"   Input lines:           {len(lines)}")
    print(f"   After noise removal:   {len(cleaned_lines)}")
    print(f"   After merge:           {len(merged_lines)}")
    print(f"   After normalization:   {len(normalized)}")
    print(f"   Relevant lines (for embedding): {len(relevant_lines)}")
    print(f"   Tagged metric lines:   {len(tagged_lines)}")
    print(f"   Final text size:       {len(cleaned_text)} chars")
    if tagged_lines:
        tag_distribution = {}
        for t in tagged_lines:
            tag_distribution[t['tag']] = tag_distribution.get(t['tag'], 0) + 1
        print(f"   Tag distribution:      {tag_distribution}")
    print("=" * 70 + "\n")
    
    return cleaned_text, tagged_lines


def process_pdf(file_bytes: bytes, chunk_size: int = 600, overlap: int = 100) -> tuple[list[str], list[dict]]:
    """
    Process PDF and return optimized chunks + tagged metrics for extraction.
    
    Parameters:
    - chunk_size: 600-800 characters for semantic chunks
    - overlap: 100 characters for context continuity
    
    Returns:
    - chunks: list[str] - For FAISS embedding
    - tagged_lines: list[dict] - Structured metric data
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    buffer = ""
    for page in reader.pages:
        text = page.extract_text() or ""
        buffer += text + " "

    # Preprocess the raw text (returns both cleaned text and tagged metrics)
    cleaned_text, tagged_lines = preprocess_pdf_text(buffer)
    
    words = cleaned_text.split()
    if not words:
        raise ValueError("PDF is unreadable or contains no extractable text after preprocessing.")

    # Character-based chunking for better semantic boundaries
    chunks = []
    text = " ".join(words)
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 50:  # Only keep meaningful chunks
            chunks.append(chunk)

    print(f"📦 Created {len(chunks)} optimized chunks (chunk_size={chunk_size}, overlap={overlap})")
    return chunks, tagged_lines


def build_index(chunks: list[str]) -> "FAISS":
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


def retrieve(vectorstore, query: str, k: int = 4) -> list[str]:
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def _map_metric_to_tag(metric_name: str) -> str:
    """Map metric name to ESG tag."""
    tag_map = {
        "ghg_scope1": "ENV_GHG",
        "energy_consumption": "ENV_ENERGY",
        "water_withdrawal": "ENV_WATER",
        "total_employees": "SOCIAL_EMP",
        "women_employees_pct": "SOCIAL_DIV",
        "board_size": "GOV_BOARD",
    }
    return tag_map.get(metric_name)


def retrieve_for_metric_with_tags(
    metric_name: str,
    tagged_lines: list[dict],
    vectorstore,
    k: int = 3,
    max_context_size: int = 500
) -> tuple[list[str], str]:
    """
    TAG-FIRST retrieval system with 8-step optimization.
    
    Returns:
    - context_lines: list[str] - Top lines for context
    - context: str - Joined context (<500 chars)
    """
    print(f"\n   🔍 TAG-BASED RETRIEVAL for {metric_name}...")
    
    # STEP 1: Tag-first filtering
    metric_tag = _map_metric_to_tag(metric_name)
    if not metric_tag or not tagged_lines:
        print(f"      ⚠️  No tagged lines available, falling back to FAISS")
        fallback_chunks = retrieve(vectorstore, METRIC_QUERIES.get(metric_name, metric_name), k=3)
        context = "\n".join(fallback_chunks)
        if len(context) > max_context_size:
            context = context[:max_context_size]
        return fallback_chunks, context
    
    # Step 1: Filter by tag
    tagged_filtered = [l for l in tagged_lines if l["tag"] == metric_tag]
    print(f"      ✓ Step 1 (Tag filter): {len(tagged_filtered)}/{len(tagged_lines)} lines")
    
    if not tagged_filtered:
        print(f"      ⚠️  No lines with tag '{metric_tag}', falling back to FAISS")
        fallback_chunks = retrieve(vectorstore, METRIC_QUERIES.get(metric_name, metric_name), k=3)
        context = "\n".join(fallback_chunks)
        if len(context) > max_context_size:
            context = context[:max_context_size]
        return fallback_chunks, context
    
    # Step 2: Keyword boosting - further filter by relevant keywords
    keywords = TAG_KEYWORDS.get(metric_tag, [])
    keyword_filtered = []
    for line in tagged_filtered:
        text_lower = line["text"].lower()
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        if keyword_count > 0 or len(line["text"]) < 200:  # Keep it if has keywords or short
            keyword_filtered.append({**line, "keyword_count": keyword_count})
    
    print(f"      ✓ Step 2 (Keyword filter): {len(keyword_filtered)} lines with relevant keywords")
    
    if not keyword_filtered:
        keyword_filtered = tagged_filtered  # Fall back to tag-filtered if no keyword matches
    
    # Step 3: Prioritize numeric lines (<200 chars)
    numeric_lines = []
    for line in keyword_filtered:
        text = line["text"]
        has_number = bool(re.search(r'\d', text))
        is_short = len(text) < 200
        
        if has_number and is_short:
            numeric_lines.append(line)
    
    print(f"      ✓ Step 3 (Numeric filter): {len(numeric_lines)} numeric lines under 200 chars")
    
    if not numeric_lines:
        numeric_lines = keyword_filtered  # Fall back if none meet criteria
    
    # Step 4: Sort by relevance (keyword count, length, number presence)
    def relevance_score(line):
        score = 0
        score += line.get("keyword_count", 0) * 10
        score -= len(line["text"]) / 10  # Shorter is better
        if re.search(r'\d', line["text"]):
            score += 5
        return score
    
    numeric_lines.sort(key=relevance_score, reverse=True)
    print(f"      ✓ Step 4 (Relevance sort): Sorted by keywords + numbers + length")
    
    # Step 5: Select top k lines
    top_lines = numeric_lines[:k]
    print(f"      ✓ Step 5 (Top-k selection): Selected top {len(top_lines)} lines")
    
    # Step 6: Build context (<500 chars)
    context_lines = []
    context = ""
    for line in top_lines:
        line_text = line["text"]
        if len(context) + len(line_text) + 2 <= max_context_size:
            context += line_text + "\n"
            context_lines.append(line_text)
    
    context = context.strip()
    print(f"      ✓ Step 6 (Context build): {len(context)} chars from {len(context_lines)} lines")
    
    # Step 7: Fallback to FAISS if no tagged context
    if not context:
        print(f"      ⚠️  Tag-based retrieval empty, falling back to FAISS")
        fallback_chunks = retrieve(vectorstore, METRIC_QUERIES.get(metric_name, metric_name), k=3)
        context = "\n".join(fallback_chunks)
        if len(context) > max_context_size:
            context = context[:max_context_size]
        return fallback_chunks, context
    
    # Step 8: Debug output
    print(f"      📍 Final context ({len(context)} chars):")
    for i, line in enumerate(context_lines[:2], 1):
        preview = line[:60] + "..." if len(line) > 60 else line
        print(f"         {i}. {preview}")
    
    return context_lines, context


# Task 6
def build_extraction_prompt(context: str) -> str:
    return f"""Extract ONLY these 5 critical ESG metrics. Return ONLY valid JSON, nothing else.

For any metric not found, use null. Return only numbers (no units).

```json
{{
  "company_name": null or string,
  "ghg_scope1": null or number,
  "total_employees": null or number,
  "women_employees_pct": null or number,
  "board_size": null or number
}}
```

CONTEXT:
{context}

Return ONLY JSON in code blocks:"""


# Task 7
def parse_metrics_from_llm_output(raw: str) -> ESGMetrics:
    try:
        if not raw or len(raw.strip()) == 0:
            print("❌ ERROR: Empty LLM response")
            return ESGMetrics()
        
        print(f"\n📝 Parsing metrics from {len(raw)} char response...")
        
        # Try different extraction methods in order
        json_str = None
        
        # Method 1: Markdown code blocks with json tag
        markdown_match = re.search(r'```json\s*\n(.*?)\n```', raw, re.DOTALL)
        if markdown_match:
            json_str = markdown_match.group(1).strip()
            print("✅ Method 1: Extracted from ```json...``` block")
        
        # Method 2: Markdown code blocks without tag
        if not json_str:
            markdown_match = re.search(r'```\s*\n(.*?)\n```', raw, re.DOTALL)
            if markdown_match:
                json_str = markdown_match.group(1).strip()
                print("✅ Method 2: Extracted from ```...``` block")
        
        # Method 3: Bare JSON object
        if not json_str:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
            if match:
                json_str = match.group()
                print("✅ Method 3: Extracted bare JSON object")
        
        if not json_str:
            print(f"❌ ERROR: No JSON found in response")
            print(f"   Response start: {raw[:300]}")
            return ESGMetrics()
        
        # Clean up the JSON string
        json_str = json_str.strip()
        if json_str.startswith('```'):
            json_str = json_str[3:].strip()
        if json_str.startswith('json'):
            json_str = json_str[4:].strip()
        if json_str.endswith('```'):
            json_str = json_str[:-3].strip()
        
        print(f"   Cleaned JSON length: {len(json_str)} chars")
        print(f"   JSON preview: {json_str[:100]}...")
        
        data = json.loads(json_str)
        print(f"✅ Successfully parsed JSON with {len(data)} keys")

        def _float(key):
            v = data.get(key)
            if v is None or v == "null" or v == "":
                return None
            try:
                v_str = str(v).strip()
                match_num = re.search(r'[\d.]+', v_str)
                if match_num:
                    return float(match_num.group())
                return None
            except (ValueError, TypeError):
                return None

        def _int(key):
            v = data.get(key)
            if v is None or v == "null" or v == "":
                return None
            try:
                v_str = str(v).strip()
                match_num = re.search(r'\d+', v_str)
                if match_num:
                    return int(match_num.group())
                return None
            except (ValueError, TypeError):
                return None

        def _str(key):
            v = data.get(key)
            if v is None or v == "null" or v == "":
                return None
            return str(v).strip() if v else None

        # Extract only the 5 critical fields
        metrics = ESGMetrics(
            company_name=_str("company_name"),
            ghg_scope1=_float("ghg_scope1"),
            total_employees=_int("total_employees"),
            women_employees_pct=_float("women_employees_pct"),
            board_size=_int("board_size"),
        )
        
        # Show what was extracted
        extracted = {k: v for k, v in {
            "company_name": metrics.company_name,
            "ghg_scope1": metrics.ghg_scope1,
            "total_employees": metrics.total_employees,
            "women_employees_pct": metrics.women_employees_pct,
            "board_size": metrics.board_size,
        }.items() if v is not None}
        
        print(f"✅ Extracted metrics: {extracted}")
        print()
        return metrics
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON - {e}")
        print(f"   Attempted JSON: {json_str[:200] if 'json_str' in locals() else 'N/A'}")
        return ESGMetrics()
    except Exception as e:
        print(f"❌ ERROR: Unexpected error - {type(e).__name__}: {e}")
        return ESGMetrics()


# Task 8
def get_llm_client():
    token = os.environ.get("GROQ_API_KEY")
    if not token:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free token at https://console.groq.com/"
        )
    return token


# Task 9.5: Metric-specific retrieval and extraction functions

METRIC_QUERIES = {
    "ghg_scope1": "Scope 1 GHG emissions direct tonnes CO2e tCO2e",
    "energy_consumption": "total energy consumption GJ MWh electricity",
    "water_withdrawal": "water withdrawal consumption KL m3",
    "total_employees": "total employees workforce headcount number staff",
    "women_employees_pct": "women employees percentage female workforce gender diversity",
    "board_size": "board size directors number governance composition",
}

# Tag-to-keywords mapping for filtering
TAG_KEYWORDS = {
    "ENV_GHG": ["scope 1", "ghg", "emissions", "co2", "tco2e", "tonnes"],
    "ENV_WATER": ["water", "withdrawal", "consumption", "kl", "m3"],
    "ENV_ENERGY": ["energy", "electricity", "kwh", "gj", "mwh", "consumption"],
    "SOCIAL_EMP": ["employee", "employees", "workforce", "headcount", "staff", "total"],
    "SOCIAL_DIV": ["women", "female", "diversity", "gender", "female", "women"],
    "GOV_BOARD": ["board", "director", "directors", "governance", "independent"],
}


def retrieve_for_metric(vectorstore, metric_name: str, k: int = 2, max_chunk_size: int = 700) -> list[str]:
    """
    Retrieve context specifically for a single metric.
    HARD LIMITS:
    - Retrieve only top 2 chunks (k=2)
    - Each chunk max 500-700 chars
    - Total context max ~1400 chars
    """
    query = METRIC_QUERIES.get(metric_name, metric_name)
    chunks = retrieve(vectorstore, query, k=k)
    
    # Enforce chunk size limits
    limited_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            limited_chunks.append(chunk)
        else:
            # Truncate chunk to max size, prefer sentences
            sentences = re.split(r'[.!?]', chunk)
            limited_text = ""
            for sent in sentences:
                if len(limited_text) + len(sent) + 2 <= max_chunk_size:
                    limited_text += sent + ". "
                else:
                    break
            if limited_text:
                limited_chunks.append(limited_text.strip())
    
    return limited_chunks


def _extract_metric_with_retry(
    metric_name: str,
    vectorstore,
    llm_token: str,
    metric_label: str,
    prompt_template: str,
    parse_func,
    tagged_lines: list[dict] = None
) -> any:
    """
    Extract a single metric with RETRY LOGIC and TAG-BASED RETRIEVAL.
    If JSON parsing fails on first try, retry once.
    """
    print(f"\n   📍 Extracting {metric_label}...")
    
    # Retrieve context using tag-based retrieval (with FAISS fallback)
    if tagged_lines:
        chunks, context = retrieve_for_metric_with_tags(
            metric_name=metric_name,
            tagged_lines=tagged_lines,
            vectorstore=vectorstore,
            k=3,
            max_context_size=500
        )
    else:
        print(f"      ⚠️  No tagged lines provided, using FAISS fallback")
        chunks = retrieve_for_metric(vectorstore, metric_name, k=2, max_chunk_size=700)
        context = "\n".join(chunks) if chunks else ""
    
    if not context:
        print(f"      ❌ No context found")
        return None
    
    total_context_size = len(context)
    print(f"      📦 Context retrieved ({total_context_size} chars)")
    
    if total_context_size > 2000:
        print(f"      ⚠️  Context exceeds 2000 chars, truncating...")
        context = context[:2000]
    
    # Build prompt
    prompt = prompt_template.format(context=context)
    
    # Try LLM call with retry
    for attempt in range(2):
        print(f"      🤖 LLM call (attempt {attempt + 1}/2, max_tokens=800, temp=0.1)...")
        
        messages = [{"role": "user", "content": prompt}]
        raw = safe_llm_call(messages, max_tokens=800, llm_token=llm_token)
        
        if raw is None:
            print(f"      ❌ LLM returned None")
            if attempt == 1:
                return None
            continue
        
        print(f"      ✅ Raw response ({len(raw)} chars): {raw[:100]}...")
        
        # Try to parse
        try:
            value = parse_func(raw)
            print(f"      ✅ Parsed value: {value}")
            return value
        except Exception as e:
            print(f"      ❌ Parse failed (attempt {attempt + 1}/2): {e}")
            if attempt == 1:
                return None
            continue
    
    return None


def _clean_json_numbers(json_str: str) -> str:
    """
    Clean JSON string by removing commas from numbers.
    Fixes cases where LLM returns "77,923" instead of "77923".
    """
    # Remove commas that appear between digits (e.g., 77,923 -> 77923)
    cleaned = re.sub(r'(\d),(\d)', r'\1\2', json_str)
    return cleaned


def extract_ghg_scope1(vectorstore, llm_token: str, tagged_lines: list[dict] = None) -> float:
    """Extract GHG Scope 1 emissions (tonnes CO2e)."""
    def parse_response(raw: str) -> float:
        raw = raw.replace("```json", "").replace("```", "").strip()
        raw = _clean_json_numbers(raw)  # Remove commas from numbers
        data = json.loads(raw)
        value = data.get("ghg_scope1")
        if value is not None:
            return float(value)
        return None
    
    prompt_template = """Extract ONLY GHG Scope 1 emissions from the context.
Return ONLY this JSON (no markdown, no explanation):
{{
"ghg_scope1": number or null
}}
Rules:
- If not found, return null
- Do NOT skip the field
- Value must be a number (tonnes CO2e)

CONTEXT:
{context}

JSON ONLY:"""
    
    return _extract_metric_with_retry(
        metric_name="ghg_scope1",
        vectorstore=vectorstore,
        llm_token=llm_token,
        metric_label="GHG Scope 1",
        prompt_template=prompt_template,
        parse_func=parse_response,
        tagged_lines=tagged_lines
    )


def extract_energy(vectorstore, llm_token: str, tagged_lines: list[dict] = None) -> float:
    """Extract energy consumption (GJ or MWh)."""
    def parse_response(raw: str) -> float:
        raw = raw.replace("```json", "").replace("```", "").strip()
        raw = _clean_json_numbers(raw)  # Remove commas from numbers
        data = json.loads(raw)
        value = data.get("energy_consumption")
        if value is not None:
            return float(value)
        return None
    
    prompt_template = """Extract ONLY total energy consumption from the context.
Return ONLY this JSON (no markdown, no explanation):
{{
"energy_consumption": number or null
}}
Rules:
- If not found, return null
- Do NOT skip the field
- Value must be a number (GJ, MWh)

CONTEXT:
{context}

JSON ONLY:"""
    
    return _extract_metric_with_retry(
        metric_name="energy_consumption",
        vectorstore=vectorstore,
        llm_token=llm_token,
        metric_label="Energy Consumption",
        prompt_template=prompt_template,
        parse_func=parse_response,
        tagged_lines=tagged_lines
    )


def extract_water(vectorstore, llm_token: str, tagged_lines: list[dict] = None) -> float:
    """Extract water withdrawal (KL or m³)."""
    def parse_response(raw: str) -> float:
        raw = raw.replace("```json", "").replace("```", "").strip()
        raw = _clean_json_numbers(raw)  # Remove commas from numbers
        data = json.loads(raw)
        value = data.get("water_withdrawal")
        if value is not None:
            return float(value)
        return None
    
    prompt_template = """Extract ONLY water withdrawal from the context.
Return ONLY this JSON (no markdown, no explanation):
{{
"water_withdrawal": number or null
}}
Rules:
- If not found, return null
- Do NOT skip the field
- Value must be a number (KL, m³)

CONTEXT:
{context}

JSON ONLY:"""
    
    return _extract_metric_with_retry(
        metric_name="water_withdrawal",
        vectorstore=vectorstore,
        llm_token=llm_token,
        metric_label="Water Withdrawal",
        prompt_template=prompt_template,
        parse_func=parse_response,
        tagged_lines=tagged_lines
    )


def extract_employees(vectorstore, llm_token: str, tagged_lines: list[dict] = None) -> int:
    """Extract total employee count."""
    def parse_response(raw: str) -> int:
        raw = raw.replace("```json", "").replace("```", "").strip()
        raw = _clean_json_numbers(raw)  # Remove commas from numbers (e.g., 77,923 -> 77923)
        data = json.loads(raw)
        value = data.get("total_employees")
        if value is not None:
            return int(float(value))
        return None
    
    prompt_template = """Extract ONLY total number of employees from the context.
Return ONLY this JSON (no markdown, no explanation):
{{
"total_employees": number or null
}}
Rules:
- If not found, return null
- Do NOT skip the field
- Value must be a whole number

CONTEXT:
{context}

JSON ONLY:"""
    
    return _extract_metric_with_retry(
        metric_name="total_employees",
        vectorstore=vectorstore,
        llm_token=llm_token,
        metric_label="Total Employees",
        prompt_template=prompt_template,
        parse_func=parse_response,
        tagged_lines=tagged_lines
    )


def extract_diversity(vectorstore, llm_token: str, tagged_lines: list[dict] = None) -> float:
    """Extract women employees percentage."""
    def parse_response(raw: str) -> float:
        raw = raw.replace("```json", "").replace("```", "").strip()
        raw = _clean_json_numbers(raw)  # Remove commas from numbers
        data = json.loads(raw)
        value = data.get("women_employees_pct")
        if value is not None:
            return float(value)
        return None
    
    prompt_template = """Extract ONLY percentage of women employees from the context.
Return ONLY this JSON (no markdown, no explanation):
{{
"women_employees_pct": number or null
}}
Rules:
- If not found, return null
- Do NOT skip the field
- Value must be 0-100

CONTEXT:
{context}

JSON ONLY:"""
    
    return _extract_metric_with_retry(
        metric_name="women_employees_pct",
        vectorstore=vectorstore,
        llm_token=llm_token,
        metric_label="Women Employees %",
        prompt_template=prompt_template,
        parse_func=parse_response,
        tagged_lines=tagged_lines
    )


def extract_board(vectorstore, llm_token: str, tagged_lines: list[dict] = None) -> int:
    """Extract board size (total directors)."""
    def parse_response(raw: str) -> int:
        raw = raw.replace("```json", "").replace("```", "").strip()
        raw = _clean_json_numbers(raw)  # Remove commas from numbers
        data = json.loads(raw)
        value = data.get("board_size")
        if value is not None:
            return int(float(value))
        return None
    
    prompt_template = """Extract ONLY board size (total number of directors) from the context.
Return ONLY this JSON (no markdown, no explanation):
{{
"board_size": number or null
}}
Rules:
- If not found, return null
- Do NOT skip the field
- Value must be a whole number

CONTEXT:
{context}

JSON ONLY:"""
    
    return _extract_metric_with_retry(
        metric_name="board_size",
        vectorstore=vectorstore,
        llm_token=llm_token,
        metric_label="Board Size",
        prompt_template=prompt_template,
        parse_func=parse_response,
        tagged_lines=tagged_lines
    )


# Task 9
EXTRACTION_QUERIES = {
    "company": "company name organization",
    "ghg_scope1": "Scope 1 GHG emissions tonnes CO2e",
    "employees": "total employees headcount workforce",
    "women": "women employees percentage female workforce",
    "board": "board size directors number",
}


def extract_esg_metrics(vectorstore, llm_token: str, tagged_lines: list[dict] = None) -> ESGMetrics:
    """
    Production-grade metric extraction using 6 dedicated functions with TAG-BASED RETRIEVAL.
    
    ARCHITECTURE:
    - Each metric: TAG-FIRST retrieval (filter by ESG tag, keyword boost, sort by relevance)
    - Each metric: FAISS fallback if no tagged lines found
    - Each metric: max_tokens=800, temperature=0.1
    - Each metric: Strict JSON format with retry logic
    - All metrics: Independent LLM calls (no truncation)
    
    HARD LIMITS:
    - Context: <500 chars per metric (tag-based) or <2000 (FAISS fallback)
    - Tokens: 800 max
    - Retries: 1 (total 2 attempts)
    """
    print("\n" + "=" * 70)
    print("🔴 PRODUCTION-GRADE METRIC-WISE EXTRACTION (TAG-BASED RETRIEVAL)")
    print("=" * 70)
    print("   Architecture: 6 independent metric functions + tag-based retrieval")
    print("   Retrieval: Tag-first filtering with FAISS fallback")
    print("   Context limit: ~500 chars per metric (tag-based)")
    print("   Token limit: 800 per metric")
    print("   Temperature: 0.1 (deterministic)")
    print("   Retry logic: Yes (1 retry on parse failure)")
    print("=" * 70)
    
    # Extract all 6 metrics independently with tag-based retrieval
    print("\n🔄 EXTRACTING 6 METRICS WITH TAG-BASED RETRIEVAL...\n")
    
    ghg_scope1 = extract_ghg_scope1(vectorstore, llm_token, tagged_lines)
    energy_consumption = extract_energy(vectorstore, llm_token, tagged_lines)
    water_withdrawal = extract_water(vectorstore, llm_token, tagged_lines)
    total_employees = extract_employees(vectorstore, llm_token, tagged_lines)
    women_employees_pct = extract_diversity(vectorstore, llm_token, tagged_lines)
    board_size = extract_board(vectorstore, llm_token, tagged_lines)
    
    # Create ESGMetrics object with all extracted values
    metrics = ESGMetrics(
        ghg_scope1=ghg_scope1,
        energy_consumption=energy_consumption,
        water_withdrawal=water_withdrawal,
        total_employees=total_employees,
        women_employees_pct=women_employees_pct,
        board_size=board_size,
    )
    
    # Summary with detailed logging
    extracted_count = sum(1 for field in [
        metrics.ghg_scope1, metrics.energy_consumption, metrics.water_withdrawal,
        metrics.total_employees, metrics.women_employees_pct, metrics.board_size
    ] if field is not None)
    
    print("\n" + "=" * 70)
    print(f"📊 EXTRACTION COMPLETE: {extracted_count}/6 metrics successfully extracted")
    print("=" * 70)
    print(f"   GHG Scope 1:          {metrics.ghg_scope1} tCO2e" if metrics.ghg_scope1 else "   GHG Scope 1:          ❌ NOT FOUND")
    print(f"   Energy Consumption:   {metrics.energy_consumption} GJ" if metrics.energy_consumption else "   Energy Consumption:   ❌ NOT FOUND")
    print(f"   Water Withdrawal:     {metrics.water_withdrawal} KL" if metrics.water_withdrawal else "   Water Withdrawal:     ❌ NOT FOUND")
    print(f"   Total Employees:      {metrics.total_employees}" if metrics.total_employees else "   Total Employees:      ❌ NOT FOUND")
    print(f"   Women Employees %:    {metrics.women_employees_pct}%" if metrics.women_employees_pct else "   Women Employees %:    ❌ NOT FOUND")
    print(f"   Board Size:           {metrics.board_size}" if metrics.board_size else "   Board Size:           ❌ NOT FOUND")
    print("=" * 70 + "\n")
    
    return metrics
