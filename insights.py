# insights.py - ESG scoring, gap detection, and insight generation

import re
import requests

from pipeline import ESGMetrics, ESGScore, Gap, InsightReport, retrieve, safe_llm_call


# Task 10
def compute_quality_bonus(fields: list[str], metrics: ESGMetrics) -> float:
    points_per_field = 30 / len(fields)
    bonus = 0.0
    for f in fields:
        val = getattr(metrics, f)
        if f == "renewable_energy_pct" and val is not None and val >= 20:
            bonus += points_per_field
        elif f == "waste_recycled_pct" and val is not None and val >= 50:
            bonus += points_per_field
        elif f == "women_employees_pct" and val is not None and val >= 30:
            bonus += points_per_field
        elif f == "training_hours_avg" and val is not None and val >= 20:
            bonus += points_per_field
        elif f == "lost_time_injury_rate" and val is not None and val <= 0.5:
            bonus += points_per_field
        elif f == "independent_directors_pct" and val is not None and val >= 50:
            bonus += points_per_field
        elif f == "women_directors_pct" and val is not None and val >= 15:
            bonus += points_per_field
        elif f == "audit_committee_independent" and val is True:
            bonus += points_per_field
        elif f == "whistleblower_policy" and val is True:
            bonus += points_per_field
        # All other fields: no quality bonus
    return min(bonus, 30.0)


# Task 11
E_FIELDS = ["ghg_scope1", "ghg_scope2", "energy_consumption",
            "renewable_energy_pct", "water_withdrawal",
            "waste_generated", "waste_recycled_pct"]

S_FIELDS = ["total_employees", "women_employees_pct",
            "training_hours_avg", "lost_time_injury_rate", "csr_spend"]

G_FIELDS = ["board_size", "independent_directors_pct",
            "women_directors_pct", "audit_committee_independent",
            "whistleblower_policy"]


def score_metrics(metrics: ESGMetrics) -> ESGScore:
    def pillar_score(fields):
        present = sum(1 for f in fields if getattr(metrics, f) is not None)
        base = (present / len(fields)) * 70
        bonus = compute_quality_bonus(fields, metrics)
        return min(base + bonus, 100.0)

    e = pillar_score(E_FIELDS)
    s = pillar_score(S_FIELDS)
    g = pillar_score(G_FIELDS)
    overall = min(max(0.4 * e + 0.35 * s + 0.25 * g, 0.0), 100.0)

    grade = "A" if overall >= 75 else "B" if overall >= 55 else "C" if overall >= 35 else "D"
    return ESGScore(environmental=e, social=s, governance=g, overall=overall, grade=grade)


# Task 12
CRITICAL_FIELDS = {
    "ghg_scope1": ("Environmental", "Scope 1 GHG emissions not disclosed — mandatory under BRSR"),
    "ghg_scope2": ("Environmental", "Scope 2 GHG emissions not disclosed — mandatory under BRSR"),
    "total_employees": ("Social", "Total employee count missing — core BRSR indicator"),
    "independent_directors_pct": ("Governance", "Board independence ratio not reported"),
}

MODERATE_FIELDS = {
    "renewable_energy_pct": ("Environmental", "Renewable energy share not disclosed"),
    "women_employees_pct": ("Social", "Gender diversity data absent"),
    "lost_time_injury_rate": ("Social", "Safety performance metrics missing"),
    "whistleblower_policy": ("Governance", "Whistleblower policy status not mentioned"),
}


def detect_gaps(metrics: ESGMetrics) -> list[Gap]:
    gaps = []
    for field, (cat, msg) in CRITICAL_FIELDS.items():
        if getattr(metrics, field) is None:
            gaps.append(Gap(cat, field, "critical", msg))
    for field, (cat, msg) in MODERATE_FIELDS.items():
        if getattr(metrics, field) is None:
            gaps.append(Gap(cat, field, "moderate", msg))
    # Minor: quality checks on present values
    if metrics.renewable_energy_pct is not None and metrics.renewable_energy_pct < 5:
        gaps.append(Gap("Environmental", "renewable_energy_pct", "minor",
                        f"Renewable energy at {metrics.renewable_energy_pct:.1f}% — well below industry average"))
    if metrics.women_directors_pct is not None and metrics.women_directors_pct < 15:
        gaps.append(Gap("Governance", "women_directors_pct", "minor",
                        "Women on board below SEBI recommended threshold of 15%"))
    return gaps


# Task 13
def build_insight_prompt(metrics, score, gaps, narrative_context: str) -> str:
    def fmt(value, suffix=""):
        if value is None:
            return "Not disclosed"
        return f"{value}{suffix}"

    gap_summary = "\n".join(f"- [{g.severity.upper()}] {g.message}" for g in gaps[:6])
    return f"""You are a senior ESG analyst. Analyze this company's ESG performance and provide actionable insights.

COMPANY: {metrics.company_name or "Unknown company"}
REPORTING YEAR: {metrics.reporting_year or "Not disclosed"}

ESG SCORES: Environmental={score.environmental:.0f}/100, Social={score.social:.0f}/100, Governance={score.governance:.0f}/100, Overall={score.overall:.0f}/100 (Grade {score.grade})

KEY METRICS:
- GHG Scope 1: {fmt(metrics.ghg_scope1, " tCO2e")}
- GHG Scope 2: {fmt(metrics.ghg_scope2, " tCO2e")}
- Renewable Energy: {fmt(metrics.renewable_energy_pct, "%")}
- Women Employees: {fmt(metrics.women_employees_pct, "%")}
- Board Independence: {fmt(metrics.independent_directors_pct, "%")}
- LTIFR: {fmt(metrics.lost_time_injury_rate)}

DISCLOSURE GAPS:
{gap_summary}

COMPANY NARRATIVE:
{narrative_context[:800]}

Provide your analysis following EXACTLY this structure with NO extra text, no explanations before/after. Bullet points are required (use *).

OUTPUT FORMAT:

HIGHLIGHTS:
* [positive finding 1]
* [positive finding 2]
* [positive finding 3]

RED FLAGS:
* [concern 1]
* [concern 2]

RECOMMENDATIONS:
* [actionable recommendation 1]
* [actionable recommendation 2]
* [actionable recommendation 3]"""


# Task 14
def parse_insight_sections(llm_response: str) -> tuple[list[str], list[str], list[str]]:
    print(f"\n📝 Parsing LLM response ({len(llm_response)} chars)...")
    print(f"   First 200 chars: {llm_response[:200]}")
    try:
        def extract_section(text: str, header: str) -> list[str]:
            # Robustly find section header and extract bullets
            pattern = rf"{header}[*:\s]*(.*?)(?=\n[A-Z_ ]+[:*]|\Z)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if not match:
                print(f"   ❌ Section not found: {header}")
                return []
            
            block = match.group(1).strip()
            items = []
            for line in block.splitlines():
                line = line.strip()
                # Remove leading bullets, dashes, or asterisks
                line = re.sub(r'^[-*•]\s*', '', line)
                if line:
                    items.append(line)
            print(f"   ✅ {header}: Extracted {len(items)} items")
            return items

        highlights = extract_section(llm_response, "HIGHLIGHTS")
        red_flags = extract_section(llm_response, "RED FLAGS")
        recommendations = extract_section(llm_response, "RECOMMENDATIONS")
        
        return highlights, red_flags, recommendations
    except Exception as e:
        print(f"   ❌ Parse Error - {e}")
        return [], [], []


# Task 15
def derive_highlights_heuristic(metrics, score) -> list[str]:
    highlights = []
    if score.environmental >= 60:
        highlights.append("Strong environmental disclosure across key metrics")
    if score.social >= 60:
        highlights.append("Solid social performance and workforce data reported")
    if score.governance >= 60:
        highlights.append("Good governance structure with board data disclosed")
    if metrics.renewable_energy_pct is not None and metrics.renewable_energy_pct >= 20:
        highlights.append(f"Renewable energy at {metrics.renewable_energy_pct:.1f}% — above industry average")
    if metrics.women_employees_pct is not None and metrics.women_employees_pct >= 30:
        highlights.append(f"Gender diversity: {metrics.women_employees_pct:.1f}% women in workforce")
    if metrics.independent_directors_pct is not None and metrics.independent_directors_pct >= 50:
        highlights.append(f"Board independence at {metrics.independent_directors_pct:.1f}% — meets best practice")
    if not highlights:
        highlights = ["ESG data partially disclosed — further reporting recommended"]
    return highlights


# Task 16
def generate_insights(metrics, vectorstore, llm_token: str) -> InsightReport:
    print("\n" + "=" * 60)
    print("INSIGHTS GENERATION STARTING")
    print("=" * 60)
    
    score = score_metrics(metrics)
    print(f"📊 Computed scores: E={score.environmental:.0f}, S={score.social:.0f}, G={score.governance:.0f}, Overall={score.overall:.0f}")
    
    gaps = detect_gaps(metrics)
    print(f"🔍 Detected {len(gaps)} disclosure gaps (C={sum(1 for g in gaps if g.severity == 'critical')}, M={sum(1 for g in gaps if g.severity == 'moderate')}, m={sum(1 for g in gaps if g.severity == 'minor')})")
    
    narrative_chunks = retrieve(vectorstore, "ESG strategy targets commitments future plans", k=4)
    narrative_context = "\n".join(narrative_chunks)
    print(f"📚 Retrieved {len(narrative_chunks)} narrative chunks ({len(narrative_context)} chars)")
    
    prompt = build_insight_prompt(metrics, score, gaps, narrative_context)
    print(f"📝 Built insight prompt ({len(prompt)} chars)")
    
    messages = [{"role": "user", "content": prompt}]
    print(f"🤖 Calling LLM for insights with llm_token={'***' if llm_token else 'None'}...")
    llm_response = safe_llm_call(messages, max_tokens=600, llm_token=llm_token)
    
    if llm_response is None:
        print("❌ INSIGHTS GENERATION FAILED: LLM returned None")
        print("   → Using rule-based fallback insights")
        llm_response = "LLM generation failed or returned None. Falling back to rule-based insights."
    else:
        print(f"✅ LLM returned insights response ({len(llm_response)} chars)")
    
    highlights, red_flags, recommendations = parse_insight_sections(llm_response)
    if not highlights:
        highlights = derive_highlights_heuristic(metrics, score)
        print(f"💡 Generated {len(highlights)} highlights from heuristics")
    else:
        print(f"💡 Parsed {len(highlights)} highlights from LLM")
        
    if not red_flags:
        red_flags = [g.message for g in gaps if g.severity == "critical"]
    if not red_flags:
        red_flags = ["No critical gaps detected"]
    print(f"🚩 Generated {len(red_flags)} red flags")
    
    if not recommendations:
        recommendations = ["Review and improve ESG disclosure completeness"]
    print(f"✅ Generated {len(recommendations)} recommendations")
    
    print("=" * 60 + "\n")
    
    return InsightReport(
        score=score,
        gaps=gaps,
        llm_insights=llm_response,
        highlights=highlights,
        red_flags=red_flags,
        recommendations=recommendations,
    )
