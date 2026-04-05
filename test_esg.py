import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pytest
from unittest.mock import patch, MagicMock

from pipeline import (
    process_pdf,
    build_index,
    retrieve,
    parse_metrics_from_llm_output,
    ESGMetrics,
    _extract_total_employee_metrics_from_tagged_lines,
    _extract_board_metrics_from_tagged_lines,
    _apply_metric_validations,
)
from insights import (
    score_metrics,
    detect_gaps,
    generate_insights,
    parse_insight_sections,
    ESGScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_reader(texts):
    """Helper to create a mock PdfReader with pages returning given texts."""
    mock_reader = MagicMock()
    pages = []
    for t in texts:
        page = MagicMock()
        page.extract_text.return_value = t
        pages.append(page)
    mock_reader.pages = pages
    return mock_reader


def make_mock_embeddings(n_chunks):
    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [[0.1] * 384] * n_chunks
    mock_emb.embed_query.return_value = [0.1] * 384
    return mock_emb


# ---------------------------------------------------------------------------
# Task 21: Tests for process_pdf
# ---------------------------------------------------------------------------

def test_process_pdf_returns_nonempty_chunks():
    text = " ".join([f"word{i}" for i in range(600)])
    with patch("pipeline.PdfReader", return_value=make_mock_reader([text])):
        chunks, tagged_lines = process_pdf(b"fake_pdf_bytes")
    assert len(chunks) > 0
    assert isinstance(tagged_lines, list)


def test_process_pdf_chunk_size():
    text = " ".join([f"word{i}" for i in range(600)])
    with patch("pipeline.PdfReader", return_value=make_mock_reader([text])):
        chunks, _ = process_pdf(b"fake_pdf_bytes", chunk_size=100, overlap=10)
    for chunk in chunks:
        assert len(chunk.split()) <= 100


def test_process_pdf_overlap():
    text = " ".join([f"word{i}" for i in range(200)])
    with patch("pipeline.PdfReader", return_value=make_mock_reader([text])):
        chunks, _ = process_pdf(b"fake_pdf_bytes", chunk_size=50, overlap=10)
    # Adjacent chunks should share words
    if len(chunks) >= 2:
        words0 = chunks[0].split()
        words1 = chunks[1].split()
        assert words0[-10:] == words1[:10]


def test_process_pdf_raises_on_empty():
    with patch("pipeline.PdfReader", return_value=make_mock_reader([""])):
        with pytest.raises(ValueError):
            process_pdf(b"fake_pdf_bytes")


# ---------------------------------------------------------------------------
# Task 22: Tests for build_index and retrieve
# ---------------------------------------------------------------------------

def _build_index_with_mock(chunks):
    """Build a FAISS index using a mock embeddings object that satisfies FAISS's
    Embeddings interface check by subclassing the base class."""
    from langchain_community.embeddings import HuggingFaceEmbeddings as _HFEmb

    class FakeEmbeddings(_HFEmb):
        def __init__(self):
            # Skip parent __init__ entirely
            pass

        def embed_documents(self, texts):
            return [[0.1] * 384 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 384

    return build_index(chunks), FakeEmbeddings()


def test_build_index_chunk_count():
    chunks = ["chunk one", "chunk two", "chunk three"]
    with patch("pipeline.HuggingFaceEmbeddings") as MockEmb:
        MockEmb.return_value = make_mock_embeddings(len(chunks))
        vs = build_index(chunks)
    assert vs.index.ntotal == len(chunks)


def test_retrieve_returns_at_most_k():
    chunks = ["alpha beta", "gamma delta", "epsilon zeta"]
    with patch("pipeline.HuggingFaceEmbeddings") as MockEmb:
        MockEmb.return_value = make_mock_embeddings(len(chunks))
        vs = build_index(chunks)
    # Mock similarity_search on the vectorstore directly
    from langchain_core.documents import Document
    vs.similarity_search = MagicMock(return_value=[
        Document(page_content=c) for c in chunks[:2]
    ])
    results = retrieve(vs, "alpha", k=2)
    assert len(results) <= 2


def test_retrieve_returns_strings():
    chunks = ["hello world", "foo bar"]
    with patch("pipeline.HuggingFaceEmbeddings") as MockEmb:
        MockEmb.return_value = make_mock_embeddings(len(chunks))
        vs = build_index(chunks)
    from langchain_core.documents import Document
    vs.similarity_search = MagicMock(return_value=[Document(page_content=chunks[0])])
    results = retrieve(vs, "hello", k=1)
    assert all(isinstance(r, str) for r in results)


# ---------------------------------------------------------------------------
# Task 23: Tests for parse_metrics_from_llm_output
# ---------------------------------------------------------------------------

ALL_KEYS = [
    "company_name", "reporting_year", "framework", "ghg_scope1", "ghg_scope2",
    "energy_consumption", "renewable_energy_pct", "water_withdrawal", "waste_generated",
    "waste_recycled_pct", "total_employees", "women_employees_pct", "training_hours_avg",
    "lost_time_injury_rate", "csr_spend", "board_size", "independent_directors_pct",
    "women_directors_pct", "audit_committee_independent", "whistleblower_policy",
]


def test_parse_all_nulls():
    data = {k: None for k in ALL_KEYS}
    raw = json.dumps(data)
    metrics = parse_metrics_from_llm_output(raw)
    assert metrics.ghg_scope1 is None
    assert metrics.total_employees is None
    assert metrics.company_name is None


def test_parse_valid_values():
    data = {k: None for k in ALL_KEYS}
    data.update({
        "company_name": "Acme Corp",
        "reporting_year": "2023",
        "ghg_scope1": 1500.5,
        "total_employees": 5000,
        "renewable_energy_pct": 25.0,
        "whistleblower_policy": True,
    })
    raw = json.dumps(data)
    metrics = parse_metrics_from_llm_output(raw)
    assert metrics.company_name == "Acme Corp"
    assert metrics.ghg_scope1 == 1500.5
    assert metrics.total_employees == 5000
    assert metrics.whistleblower_policy is True


def test_parse_malformed_returns_default():
    metrics = parse_metrics_from_llm_output("not json at all !!!")
    assert isinstance(metrics, ESGMetrics)
    assert metrics.ghg_scope1 is None


def test_employee_table_parser_prefers_main_total_row():
    tagged_lines = [
        {"tag": "SOCIAL_EMP", "text": "3 Total employees (D E) 27 21 77.8% 6 22.2%"},
        {"tag": "SOCIAL_EMP", "text": "3 Total employees (D E) 75,323 55,752 74.0% 19,571 26.0%"},
    ]
    total, women_pct = _extract_total_employee_metrics_from_tagged_lines(tagged_lines)
    assert total == 75323
    assert women_pct == 26.0


def test_board_table_parser_reads_board_size_and_women_pct():
    tagged_lines = [
        {"tag": "GOV_BOARD", "text": "No. (B) % (B/A) Board of Directors 11 3 27.3%"},
    ]
    board_size, women_pct = _extract_board_metrics_from_tagged_lines(tagged_lines)
    assert board_size == 11
    assert women_pct == 27.3


def test_metric_validation_rejects_weak_llm_values():
    metrics = ESGMetrics(
        ghg_scope1=10.2,
        energy_consumption=23.5,
        women_employees_pct=77.8,
        total_employees=27,
        independent_directors_pct=27.3,
    )
    tagged_lines = [
        {"tag": "SOCIAL_EMP", "text": "Name of the Listed Entity Kotak Mahindra Bank Limited"},
        {"tag": "SOCIAL_EMP", "text": "3 Total employees (D E) 75,323 55,752 74.0% 19,571 26.0%"},
        {"tag": "GOV_BOARD", "text": "Board of Directors 11 3 27.3%"},
    ]
    validated = _apply_metric_validations(metrics, tagged_lines)
    assert validated.total_employees == 75323
    assert validated.women_employees_pct == 26.0
    assert validated.ghg_scope1 is None
    assert validated.energy_consumption is None
    assert validated.company_name == "Kotak Mahindra Bank Limited"
    assert validated.independent_directors_pct is None


# ---------------------------------------------------------------------------
# Task 24: Tests for score_metrics
# ---------------------------------------------------------------------------

def test_score_all_none_is_grade_d():
    metrics = ESGMetrics()
    score = score_metrics(metrics)
    assert score.grade == "D"
    assert 0.0 <= score.environmental <= 100.0
    assert 0.0 <= score.social <= 100.0
    assert 0.0 <= score.governance <= 100.0
    assert 0.0 <= score.overall <= 100.0


def test_score_grade_thresholds():
    metrics = ESGMetrics(
        ghg_scope1=100.0, ghg_scope2=50.0, energy_consumption=1000.0,
        renewable_energy_pct=25.0, water_withdrawal=500.0, waste_generated=10.0,
        waste_recycled_pct=60.0,
        total_employees=1000, women_employees_pct=35.0, training_hours_avg=25.0,
        lost_time_injury_rate=0.3, csr_spend=5.0,
        board_size=10, independent_directors_pct=55.0, women_directors_pct=20.0,
        audit_committee_independent=True, whistleblower_policy=True,
    )
    score = score_metrics(metrics)
    assert score.grade in {"A", "B"}
    assert score.overall >= 55.0


def test_score_bounds_always_valid():
    for _ in range(5):
        metrics = ESGMetrics(ghg_scope1=100.0, total_employees=500)
        score = score_metrics(metrics)
        assert 0.0 <= score.overall <= 100.0
        assert score.grade in {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Task 25: Tests for detect_gaps
# ---------------------------------------------------------------------------

def test_critical_gaps_before_moderate():
    metrics = ESGMetrics()  # all None
    gaps = detect_gaps(metrics)
    severities = [g.severity for g in gaps]
    critical_idx = [i for i, s in enumerate(severities) if s == "critical"]
    moderate_idx = [i for i, s in enumerate(severities) if s == "moderate"]
    if critical_idx and moderate_idx:
        assert max(critical_idx) < min(moderate_idx)


def test_all_critical_fields_none_produces_critical_gaps():
    metrics = ESGMetrics()
    gaps = detect_gaps(metrics)
    critical_fields = {"ghg_scope1", "ghg_scope2", "total_employees", "independent_directors_pct"}
    gap_fields = {g.field for g in gaps if g.severity == "critical"}
    assert critical_fields == gap_fields


def test_minor_gap_low_renewable():
    metrics = ESGMetrics(renewable_energy_pct=2.0)
    gaps = detect_gaps(metrics)
    minor_gaps = [g for g in gaps if g.severity == "minor" and g.field == "renewable_energy_pct"]
    assert len(minor_gaps) == 1


def test_minor_gap_low_women_directors():
    metrics = ESGMetrics(women_directors_pct=10.0)
    gaps = detect_gaps(metrics)
    minor_gaps = [g for g in gaps if g.severity == "minor" and g.field == "women_directors_pct"]
    assert len(minor_gaps) == 1


def test_no_gaps_when_all_present_and_above_threshold():
    metrics = ESGMetrics(
        ghg_scope1=100.0, ghg_scope2=50.0, total_employees=1000,
        independent_directors_pct=55.0, renewable_energy_pct=25.0,
        women_employees_pct=35.0, lost_time_injury_rate=0.3,
        whistleblower_policy=True, women_directors_pct=20.0,
    )
    gaps = detect_gaps(metrics)
    critical_moderate = [g for g in gaps if g.severity in ("critical", "moderate")]
    assert len(critical_moderate) == 0


# ---------------------------------------------------------------------------
# Task 26: Tests for parse_insight_sections
# ---------------------------------------------------------------------------

def test_parse_well_formed_response():
    response = """HIGHLIGHTS:
- Strong renewable energy usage
- Good board diversity

RED FLAGS:
- Missing GHG data

RECOMMENDATIONS:
- Improve water disclosure
- Set science-based targets

ANALYSIS:
Some analysis here."""
    highlights, red_flags, recommendations = parse_insight_sections(response)
    assert "Strong renewable energy usage" in highlights
    assert "Missing GHG data" in red_flags
    assert "Improve water disclosure" in recommendations


def test_parse_malformed_returns_empty_lists():
    highlights, red_flags, recommendations = parse_insight_sections("garbage text with no sections")
    assert highlights == []
    assert red_flags == []
    assert recommendations == []


def test_parse_partial_response():
    response = """HIGHLIGHTS:
- One highlight

ANALYSIS:
Some text."""
    highlights, red_flags, recommendations = parse_insight_sections(response)
    assert len(highlights) == 1
    assert red_flags == []
    assert recommendations == []


# ---------------------------------------------------------------------------
# Task 27: Integration smoke test
# ---------------------------------------------------------------------------

def test_full_pipeline_smoke():
    """Integration test: process_pdf → build_index → extract_esg_metrics → generate_insights"""
    text = " ".join([f"word{i}" for i in range(1000)])

    extraction_json = json.dumps({
        "company_name": "Test Corp", "reporting_year": "2023", "framework": "BRSR",
        "ghg_scope1": 1000.0, "ghg_scope2": 500.0, "energy_consumption": 5000.0,
        "renewable_energy_pct": 22.0, "water_withdrawal": 1000.0, "waste_generated": 50.0,
        "waste_recycled_pct": 55.0, "total_employees": 2000, "women_employees_pct": 32.0,
        "training_hours_avg": 22.0, "lost_time_injury_rate": 0.4, "csr_spend": 10.0,
        "board_size": 12, "independent_directors_pct": 58.0, "women_directors_pct": 25.0,
        "audit_committee_independent": True, "whistleblower_policy": True,
    })
    insight_response = """HIGHLIGHTS:
- Strong renewable energy
- Good governance

RED FLAGS:
- Minor water disclosure gap

RECOMMENDATIONS:
- Set net-zero targets

ANALYSIS:
Test Corp shows strong ESG performance."""

    # Mock the Groq client call for LLM API
    mock_client_instance = MagicMock()
    mock_chat_response1 = MagicMock()
    mock_chat_response1.choices = [MagicMock(message=MagicMock(content=extraction_json))]
    mock_chat_response2 = MagicMock()
    mock_chat_response2.choices = [MagicMock(message=MagicMock(content=insight_response))]
    
    mock_client_instance.chat.completions.create.side_effect = [
        mock_chat_response1,
        mock_chat_response2
    ]

    from langchain_core.documents import Document
    from pipeline import extract_esg_metrics

    with patch("pipeline.PdfReader", return_value=make_mock_reader([text])), \
         patch("pipeline.HuggingFaceEmbeddings") as MockEmb, \
         patch("pipeline.Groq", return_value=mock_client_instance):
        chunks, tagged_lines = process_pdf(b"fake")
        MockEmb.return_value = make_mock_embeddings(len(chunks))
        vs = build_index(chunks)

    # Mock similarity_search so retrieve works without real embeddings
    dummy_docs = [Document(page_content=chunks[0])] * 3
    vs.similarity_search = MagicMock(return_value=dummy_docs)

    metrics = extract_esg_metrics(vs, "fake_token", tagged_lines)
    report = generate_insights(metrics, vs, "fake_token")

    assert len(report.llm_insights.strip()) > 0
    assert 1 <= len(report.highlights) <= 5
    assert 1 <= len(report.recommendations) <= 5
    assert report.score.grade in {"A", "B", "C", "D"}
