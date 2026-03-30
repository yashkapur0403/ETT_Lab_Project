# app.py - Streamlit application entry point

import os
import sys

# Crucial fix for Windows ASCII bugs with Hugging Face Hub (which uses ✅ in logs)
os.environ["PYTHONUTF8"] = "1" 
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import streamlit as st

from pipeline import ESGMetrics, ESGScore, InsightReport, process_pdf, build_index, get_llm_client, extract_esg_metrics
from insights import generate_insights


# Task 17
def render_metrics_panel(metrics: ESGMetrics) -> None:
    if metrics.company_name or metrics.reporting_year:
        parts = []
        if metrics.company_name:
            parts.append(f"**Company:** {metrics.company_name}")
        if metrics.reporting_year:
            parts.append(f"**Reporting Year:** {metrics.reporting_year}")
        st.markdown("  |  ".join(parts))

    def fmt(val, suffix=""):
        if val is None:
            return "Not disclosed"
        if isinstance(val, float):
            return f"{val:.1f}{suffix}"
        return f"{val}{suffix}"

    def fmt_bool(val):
        if val is None:
            return "Not disclosed"
        return "Yes" if val else "No"

    env_tab, soc_tab, gov_tab = st.tabs(["🌿 Environmental", "👥 Social", "🏛️ Governance"])

    with env_tab:
        c1, c2, c3 = st.columns(3)
        c1.metric("GHG Scope 1 (tCO₂e)", fmt(metrics.ghg_scope1))
        c2.metric("GHG Scope 2 (tCO₂e)", fmt(metrics.ghg_scope2))
        c3.metric("Energy Consumption (GJ)", fmt(metrics.energy_consumption))
        c4, c5, c6 = st.columns(3)
        c4.metric("Renewable Energy (%)", fmt(metrics.renewable_energy_pct, "%"))
        c5.metric("Water Withdrawal (KL)", fmt(metrics.water_withdrawal))
        c6.metric("Waste Generated (MT)", fmt(metrics.waste_generated))
        c7, _, _ = st.columns(3)
        c7.metric("Waste Recycled (%)", fmt(metrics.waste_recycled_pct, "%"))

    with soc_tab:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Employees", fmt(metrics.total_employees))
        c2.metric("Women Employees (%)", fmt(metrics.women_employees_pct, "%"))
        c3.metric("Avg Training Hours", fmt(metrics.training_hours_avg))
        c4, c5, _ = st.columns(3)
        c4.metric("LTIFR", fmt(metrics.lost_time_injury_rate))
        c5.metric("CSR Spend (₹ Cr)", fmt(metrics.csr_spend))

    with gov_tab:
        c1, c2, c3 = st.columns(3)
        c1.metric("Board Size", fmt(metrics.board_size))
        c2.metric("Independent Directors (%)", fmt(metrics.independent_directors_pct, "%"))
        c3.metric("Women Directors (%)", fmt(metrics.women_directors_pct, "%"))
        c4, c5, _ = st.columns(3)
        c4.metric("Audit Committee Independent", fmt_bool(metrics.audit_committee_independent))
        c5.metric("Whistleblower Policy", fmt_bool(metrics.whistleblower_policy))


# Task 18
def render_score_panel(score: ESGScore) -> None:
    grade_messages = {
        "A": "Excellent ESG disclosure and performance — industry leader.",
        "B": "Good ESG performance with room for improvement in some areas.",
        "C": "Moderate ESG disclosure — significant gaps remain.",
        "D": "Poor ESG disclosure — major improvements needed across all pillars.",
    }

    st.metric("Overall ESG Grade", score.grade, delta=f"{score.overall:.1f}/100")

    grade_fn = {"A": st.success, "B": st.info, "C": st.warning, "D": st.error}.get(score.grade, st.info)
    grade_fn(grade_messages.get(score.grade, ""))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🌿 Environmental", f"{score.environmental:.1f}/100")
        st.progress(int(score.environmental) / 100)
    with c2:
        st.metric("👥 Social", f"{score.social:.1f}/100")
        st.progress(int(score.social) / 100)
    with c3:
        st.metric("🏛️ Governance", f"{score.governance:.1f}/100")
        st.progress(int(score.governance) / 100)


# Task 19
def render_insights_panel(report: InsightReport) -> None:
    if not report:
        st.error("❌ Failed to generate insights report")
        return
    
    st.markdown("### ✅ Highlights")
    if report.highlights and isinstance(report.highlights, list) and len(report.highlights) > 0:
        for h in report.highlights:
            st.success(h)
    else:
        st.info("No highlights available")

    st.markdown("### 🚩 Red Flags")
    if report.red_flags and isinstance(report.red_flags, list) and len(report.red_flags) > 0:
        for r in report.red_flags:
            st.error(r)
    else:
        st.info("No risks detected")

    st.markdown("### 💡 Recommendations")
    if report.recommendations and isinstance(report.recommendations, list) and len(report.recommendations) > 0:
        for r in report.recommendations:
            st.info(r)
    else:
        st.info("No recommendations available")

    st.markdown("### 🔍 Disclosure Gaps")
    if report.gaps and len(report.gaps) > 0:
        severity_fn = {"critical": st.error, "moderate": st.warning, "minor": st.info}
        for gap in report.gaps:
            fn = severity_fn.get(gap.severity, st.info)
            fn(f"[{gap.severity.upper()}] {gap.category} — {gap.message}")
    else:
        st.success("No disclosure gaps detected.")

    if report.llm_insights:
        with st.expander("📄 Full AI Analysis"):
            st.markdown(report.llm_insights)


# Task 20
def main():
    st.set_page_config(
        page_title="ESG Report Analyzer",
        page_icon="🌱",
        layout="wide"
    )

    st.title("🌱 AI-Driven BRSR / ESG Report Analyzer")
    st.caption("Upload an ESG or BRSR PDF report to extract metrics, score performance, and generate AI-powered insights.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.text_input("Groq API Key", type="password",
                                  help="Get a free key at https://console.groq.com/")
        if groq_api_key:
            if st.session_state.get("GROQ_API_KEY") != groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
                st.session_state["GROQ_API_KEY"] = groq_api_key
                st.rerun()
        st.divider()
        st.markdown("**Supported Frameworks:** BRSR, GRI, TCFD, SASB")
        st.markdown("**Models:** mixtral-8x7b, llama-3.1-70b, llama-3.1-8b (Auto-selected)")

    uploaded = st.file_uploader(
        "Upload ESG/BRSR PDF Report",
        type=["pdf"],
        help="Upload a PDF ESG or BRSR report to analyze"
    )

    if uploaded is not None:
        st.divider()
        
        # Prevent LLM call until key is present
        api_key = st.session_state.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.warning("Please enter Groq API key to enable AI insights")
            st.stop()
            
        print("GROQ KEY:", "****" + str(api_key)[-4:] if api_key else "None")
        print("LLM CALL TRIGGERED")

        # Step 1: Process PDF
        with st.spinner("📄 Extracting and chunking document..."):
            try:
                chunks, tagged_lines = process_pdf(uploaded.read())
                st.success(f"✅ Document processed — {len(chunks)} chunks indexed, {len(tagged_lines)} metrics tagged")
            except ValueError as e:
                st.error(f"❌ {e}")
                st.stop()

        # Step 2: Build index
        with st.spinner("🔍 Building semantic search index..."):
            vectorstore = build_index(chunks)

        # Step 3: Init LLM
        with st.spinner("🤖 Connecting to AI model..."):
            try:
                llm_token = get_llm_client()
            except ValueError as e:
                st.error(f"❌ {e}")
                st.stop()

        # Step 4: Extract metrics
        with st.spinner("📊 Extracting ESG metrics from report..."):
            metrics = extract_esg_metrics(vectorstore, llm_token, tagged_lines)
            # Check if any key metrics were actually extracted
            has_data = any([
                metrics.ghg_scope1 is not None,
                metrics.ghg_scope2 is not None,
                metrics.total_employees is not None,
                metrics.board_size is not None,
                metrics.company_name is not None
            ])
            if not has_data:
                st.warning("⚠️ **Limited data extracted**: The report may not contain structured ESG metrics, or LLM extraction encountered issues. Results below are based on available data.")
            else:
                st.success(f"✅ Extracted metrics for {metrics.company_name or 'company'}")

        # Step 5: Generate insights
        with st.spinner("💡 Generating AI-powered insights..."):
            report = generate_insights(metrics, vectorstore, llm_token)
            if "LLM generation failed" in report.llm_insights or "Falling back" in report.llm_insights:
                st.warning("⚠️ **Using rule-based insights**: AI model unavailable, displaying insights based on extracted metrics.")
            else:
                st.success("✅ AI insights generated successfully")

        st.success("✅ Analysis complete!")
        st.divider()

        # Render dashboard
        st.subheader("📊 ESG Metrics")
        render_metrics_panel(metrics)

        st.divider()
        st.subheader("🏆 ESG Performance Score")
        render_score_panel(report.score)

        st.divider()
        st.subheader("💡 AI-Generated Insights")
        render_insights_panel(report)


if __name__ == "__main__":
    main()
