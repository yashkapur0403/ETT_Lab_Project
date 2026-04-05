# AI-Driven BRSR / ESG Report Analyzer

An AI-powered ESG and BRSR report analysis tool built with Streamlit, Retrieval-Augmented Generation (RAG), semantic search, and Groq-hosted LLMs.

The project allows a user to upload an ESG or BRSR PDF report, extract structured ESG metrics, compute ESG scores, identify disclosure gaps, and generate AI-based insights such as highlights, red flags, and recommendations.

---

## Features

- Upload ESG / BRSR PDF reports through a Streamlit UI
- Extract structured ESG metrics from PDF reports
- Use semantic retrieval with FAISS for focused context selection
- Apply a RAG-based pipeline for ESG analysis
- Compute Environmental, Social, and Governance scores
- Generate an overall ESG grade
- Detect disclosure gaps across ESG pillars
- Generate AI-powered:
  - Highlights
  - Red flags
  - Recommendations
- Use rule-based validation and fallback logic to improve extraction quality
- Display results in an interactive dashboard

---

## Project Overview

This project is designed to analyze Business Responsibility and Sustainability Reports (BRSR) and ESG disclosures in PDF format.

### Workflow

1. User uploads a PDF report  
2. PDF text is extracted and preprocessed  
3. Relevant ESG lines are tagged and cleaned  
4. Text is chunked for semantic retrieval  
5. FAISS vector index is built using sentence-transformer embeddings  
6. Relevant ESG context is retrieved  
7. Groq-hosted LLMs extract structured ESG metrics  
8. Validation logic corrects or rejects weak extractions  
9. ESG scores and disclosure gaps are generated  
10. AI-generated insights are shown in the Streamlit dashboard  

---

## Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### PDF Processing
- pypdf

### Embeddings and Retrieval
- LangChain Community FAISS  
- HuggingFaceEmbeddings  
- sentence-transformers/all-MiniLM-L6-v2  

### LLM Inference
- Groq Python SDK  
- Groq-hosted open models:
  - mixtral-8x7b-32768  
  - llama-3.1-70b-versatile  
  - llama-3.1-8b-instant  
  - llama2-70b-4096  

### Testing
- pytest

---

## Project Structure

```text
brsr_esg_analyzer/
│
├── app.py
├── pipeline.py
├── insights.py
├── requirements.txt
├── test_esg.py
└── README.md
```

### File Descriptions

**app.py**
- Streamlit application entry point  
- Handles UI, upload flow, and dashboard rendering  

**pipeline.py**
- Core RAG pipeline  
- PDF processing, preprocessing, chunking, retrieval, extraction, validation  

**insights.py**
- ESG scoring logic  
- Gap detection  
- Insight prompt creation and parsing  

**requirements.txt**
- Project dependencies  

**test_esg.py**
- Unit and integration tests  

---

## Architecture

This project uses a RAG-oriented architecture for ESG extraction and insight generation.

### High-Level Flow

- PDF Upload (Streamlit)  
- PDF Text Extraction (pypdf)  
- Text Preprocessing and ESG Tagging  
- Chunking  
- Embeddings + FAISS Index  
- Semantic Retrieval  
- LLM-Based Metric Extraction  
- Rule-Based Validation  
- ESG Scoring  
- Narrative Retrieval  
- Insight Generation  
- Dashboard Output  

---

## ESG Metrics Extracted

### Environmental
- GHG Scope 1  
- GHG Scope 2  
- Energy consumption  
- Renewable energy percentage  
- Water withdrawal  
- Waste generated  
- Waste recycled percentage  

### Social
- Total employees  
- Women employees percentage  
- Average training hours  
- Lost Time Injury Rate (LTIFR)  
- CSR spend  

### Governance
- Board size  
- Independent directors percentage  
- Women directors percentage  
- Audit committee independent  
- Whistleblower policy  

### Metadata
- Company name  
- Reporting year  
- Reporting framework  

---

## ESG Scoring

The application computes scores for:

- Environmental  
- Social  
- Governance  

These are combined into an overall ESG score and grade.

### Grade Bands

- **A** : Excellent ESG disclosure and performance  
- **B** : Good ESG performance  
- **C** : Moderate ESG disclosure  
- **D** : Poor ESG disclosure  

### Scoring Logic

- Metric availability  
- Quality and threshold bonuses  
- Weighted ESG pillar aggregation  

---

## Disclosure Gap Detection

The project detects:

- Critical disclosure gaps  
- Moderate disclosure gaps  
- Minor improvement areas  

### Examples

- Missing Scope 1 / Scope 2 emissions  
- Missing employee counts  
- Missing board independence  
- Missing renewable energy disclosure  
- Missing whistleblower policy disclosure  

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd brsr_esg_analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

```bash
python -m streamlit run app.py
```

Then:

- Open the local Streamlit URL in your browser  
- Enter your Groq API key in the sidebar  
- Upload an ESG / BRSR PDF report  
- Review extracted metrics, scores, and insights  

---

## Running Tests

```bash
pytest -q
```

---

## Example Dummy Report Testing

A clean, label-based dummy ESG report can be used to validate the current pipeline.

### Best Performance Conditions

- Metrics written clearly as `Label: Value`  
- No dense multi-column tables  
- Subcategories spaced apart  
- Units explicitly provided  

### Example

```text
Company Name: Example Company Pvt Ltd
Reporting Year: FY 2024-25
Framework: BRSR

Scope 1 GHG emissions: 1250 tCO2e
Scope 2 GHG emissions: 980 tCO2e
Total energy consumption: 18500 GJ
Renewable energy: 24%
Water withdrawal: 263790 KL
Waste generated: 120 MT
Waste recycled: 62%

Total employees: 75323
Women employees: 26.0%
Average training hours: 23.5
Lost Time Injury Rate: 0.18
CSR spend: 48.2 crore

Board size: 11
Independent directors: 54.5%
Women directors: 27.3%
Audit committee independent: Yes
Whistleblower policy: Yes
```

---

## Current Strengths

- Clean Streamlit UI  
- End-to-end RAG pipeline  
- Structured ESG extraction  
- FAISS-based semantic retrieval  
- Hybrid LLM + validation approach  
- ESG scoring and gap detection  
- Automated testing support  
- Strong performance on clean reports  

---

## Current Limitations

- Complex PDFs with dense tables may cause extraction errors  
- Governance/boolean fields are less stable  
- Accuracy depends on PDF structure  
- Scoring focuses on disclosure completeness (not benchmarking)  
- Embeddings depend on Hugging Face availability  
- Better performance on clean vs complex reports  

---

## Future Improvements

- Better table parsing  
- Metric-specific extraction rules  
- Source-line traceability  
- Improved governance extraction  
- Confidence scores  
- Better fallback handling  
- Support for more ESG formats  
- Industry benchmarking  

---

## Security Notes

Do NOT commit:

- API keys  
- `.venv/`  
- `__pycache__/`  
- `.pytest_cache/`  
- Uploaded confidential PDFs  
- Local secret/token files  

Use a proper `.gitignore` to exclude these.
