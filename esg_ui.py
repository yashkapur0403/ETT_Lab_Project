import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Sustainability Report Analyzer",
    layout="wide"
)

# ---------------- Sidebar ----------------
st.sidebar.title("AI ESG Analyzer")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Upload Report", "Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.info("This is the UI prototype version.\n\nAI integration coming soon.")

# ---------------- Upload Page ----------------
if page == "Upload Report":

    st.title("Sustainability Report Upload")

    st.markdown("""
    Upload a Business Responsibility or Sustainability report (PDF).
    The system will analyze ESG metrics and generate structured insights.
    """)

    uploaded_file = st.file_uploader("Upload PDF Report", type=["pdf"])

    st.markdown("---")

    st.subheader("Report Processing Workflow")

    st.markdown("""
    1. Document ingestion  
    2. Content extraction  
    3. Data structuring  
    4. AI-based analysis  
    5. Structured output generation  
    """)

    if uploaded_file:
        st.success("Report uploaded successfully (analysis module not connected yet).")

# ---------------- Dashboard Page ----------------
if page == "Analytics Dashboard":

    st.title("ESG Analytics Dashboard")

    st.markdown("Prototype visualization based on sample structured ESG output.")

    # Sample Mock Data (Temporary)
    data = {
        "Category": ["Environment", "Social", "Governance"],
        "Score": [75, 60, 80]
    }

    df = pd.DataFrame(data)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df, x="Category", y="Score",
                     title="ESG Score Overview",
                     range_y=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(df, names="Category", values="Score",
                      title="ESG Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Key Insights (Sample Output)")

    st.markdown("""
    - Environmental reporting is strong with detailed disclosures.  
    - Social metrics show moderate diversity representation.  
    - Governance structure indicates balanced board independence.  
    """)

    st.info("Note: These values are placeholders for UI demonstration purposes.")
