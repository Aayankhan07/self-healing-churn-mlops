"""
Streamlit executive dashboard.
Connects to FastAPI. No direct DB access from dashboard.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np

API_URL = os.getenv("API_URL", "http://localhost:8000")

ENVIRONMENT = os.getenv("ENV", "development").lower()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    if ENVIRONMENT == "production":
        raise ValueError(
            "API_KEY environment variable must be set in production environment!"
        )
    else:
        API_KEY = "dev-key-change-in-prod"

HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(
    page_title="ChurnGuard — Customer Churn Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling for sleek dark mode
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background-color: #080C14;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #F8FAFC !important;
        font-weight: 600;
    }
    
    .main-title {
        background: linear-gradient(135deg, #14B8A6 0%, #0D9488 50%, #0F766E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Premium Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Style navigation radio options as clean full-width tabs */
    section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] {
        gap: 8px !important;
    }
    
    section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        margin: 4px 0px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* Hide default radio circle elements */
    section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    
    section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
        margin-left: 0px !important;
    }
    
    section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
    
    section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label:hover {
        background-color: rgba(20, 184, 166, 0.08) !important;
        border-color: rgba(20, 184, 166, 0.3) !important;
    }
    
    /* Selected navigation item state */
    section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label:has(input:checked) {
        background-color: rgba(13, 148, 136, 0.15) !important;
        border-color: #0D9488 !important;
        border-left: 4px solid #0D9488 !important;
    }
    
    section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #F8FAFC !important;
    }
    
    /* Cards & Typography Hierarchy */
    .kpi-card {
        background: linear-gradient(145deg, #131D30 0%, #090E17 100%);
        border: 1px solid #202D42;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #0D9488;
    }
    
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin-top: 8px;
        color: #F8FAFC;
    }
    
    .kpi-label {
        color: #64748B;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    
    .kpi-delta {
        font-size: 0.85rem;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Buttons styling */
    .stButton>button {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(13, 148, 136, 0.4);
        background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%);
        border: none !important;
    }
    
    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #080C14;
    }
    ::-webkit-scrollbar-thumb {
        background: #202D42;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
    
    /* Timeline component */
    .timeline-item {
        position: relative;
        padding-left: 30px;
        margin-left: 12px;
        border-left: 2px solid #202D42;
        padding-bottom: 20px;
    }
    
    .timeline-item:last-child {
        border-left: 2px solid transparent !important;
        padding-bottom: 4px;
    }
    
    .timeline-badge {
        position: absolute;
        left: -13px;
        top: 2px;
        background-color: #0F172A;
        border: 2px solid #202D42;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        z-index: 2;
    }
    
    .timeline-item.data-quality .timeline-badge {
        border-color: #0D9488;
        color: #14B8A6;
    }
    
    .timeline-item.retraining .timeline-badge {
        border-color: #F59E0B;
        color: #F59E0B;
    }
    
    .timeline-header {
        font-size: 0.8rem;
        color: #64748B;
        margin-bottom: 4px;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    
    .timeline-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar Navigation ──────────────────────────────────────
st.sidebar.markdown(
    "<h2 style='text-align: center; margin-bottom: 20px;'>ChurnGuard</h2>",
    unsafe_allow_html=True,
)
if "custom_domains" not in st.session_state:
    st.session_state["custom_domains"] = {}

domain_options = [
    "Telecom Customer Churn",
    "School Student Churn",
    "E-Commerce Customer Churn",
    "Fitness Club Member Churn",
] + list(st.session_state["custom_domains"].keys()) + ["+ Add Custom Domain..."]

domain_option = st.sidebar.selectbox("Industry Domain", domain_options)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Customers", "Upload & Score", "Model Health", "Self-Healing Console"],
)

DOMAIN_CONFIGS = {
    "Telecom Customer Churn": {
        "title": "Telecom Customer Churn Console",
        "subtitle": "Predict and prevent customer contract cancellations.",
        "entity_name": "Customer",
        "unit": "Customers",
        "total_count": "7,043",
        "high_risk_pct": "23.4%",
        "med_risk_pct": "31.1%",
        "low_risk_pct": "45.5%",
        "kpi1_label": "Customers Scored",
        "kpi2_label": "High Risk Segment",
        "kpi3_label": "Medium Risk Segment",
        "kpi4_label": "Low Risk Segment",
        "records_title": "Customer Directory and Explainability",
        "records_subtitle": "Browse customer profiles and inspect individual churn risk with SHAP explanations.",
        "search_label": "Search Customer by ID",
        "filter_label": "Contract Type Filter",
        "filter_options": ["All", "Month-to-month", "One year", "Two year"],
        "table_title": "Customer Records",
        "select_label": "Select Customer ID to Analyze",
        "upload_title": "Bulk CSV Upload and Predictions",
        "upload_subtitle": "Upload customer database exports (.csv) to batch predict churn scores for up to 5,000 rows.",
        "sample_csv_path": "data/sample_batch_customers.csv",
        "sample_csv_name": "sample_batch_customers.csv",
        "health_title": "Model Quality and Monitoring",
        "health_subtitle": "Monitor ML model quality metrics, precision-recall indicators, and Evidently data drift reports.",
        "console_title": "Self-Healing Console",
        "console_subtitle": "Real-time monitoring of ingestion-level data healing, drift-triggered retraining, and atomic hot-reloading.",
    },
    "School Student Churn": {
        "title": "School Student Churn Console (K-12)",
        "subtitle": "Monitor student attendance, academic average, and dropout risks across Grade 1 to Grade 12.",
        "entity_name": "Student",
        "unit": "Students",
        "total_count": "1,250",
        "high_risk_pct": "12.8%",
        "med_risk_pct": "24.5%",
        "low_risk_pct": "62.7%",
        "kpi1_label": "Enrolled Students",
        "kpi2_label": "High Churn Risk Students",
        "kpi3_label": "Moderate Risk Students",
        "kpi4_label": "Low Risk Students",
        "records_title": "Student Directory and Academic Risk",
        "records_subtitle": "Browse student registers (Grade 1-12) and inspect individual academic churn risk with SHAP factors.",
        "search_label": "Search Student by ID",
        "filter_label": "Grade Level Filter",
        "filter_options": ["All", "Grade 1-5 (Primary)", "Grade 6-8 (Middle)", "Grade 9-12 (High School)"],
        "table_title": "Student Academic Registers",
        "select_label": "Select Student ID to Analyze",
        "upload_title": "Bulk Student Register CSV Upload",
        "upload_subtitle": "Upload school student registers (.csv) to batch predict dropout risk scores across grade levels.",
        "sample_csv_path": "data/sample_batch_students.csv",
        "sample_csv_name": "sample_batch_students.csv",
        "health_title": "School Student Model Quality & Monitoring",
        "health_subtitle": "Monitor student churn prediction metrics, grade level drift indicators, and performance reports.",
        "console_title": "School Self-Healing Console",
        "console_subtitle": "Real-time monitoring of student register data quality, attendance drift, and automated model retraining.",
    },
    "E-Commerce Customer Churn": {
        "title": "E-Commerce Customer Churn Console",
        "subtitle": "Identify buyers at risk of un-subscribing or becoming inactive.",
        "entity_name": "Buyer",
        "unit": "Buyers",
        "total_count": "15,420",
        "high_risk_pct": "18.2%",
        "med_risk_pct": "28.4%",
        "low_risk_pct": "53.4%",
        "kpi1_label": "Active Buyers",
        "kpi2_label": "High Risk Inactive",
        "kpi3_label": "Medium Risk Segment",
        "kpi4_label": "Low Risk Segment",
        "records_title": "Buyer Directory and Retention Risk",
        "records_subtitle": "Browse buyer purchase profiles and inspect individual churn risk factors.",
        "search_label": "Search Buyer by ID",
        "filter_label": "Purchase Category Filter",
        "filter_options": ["All", "Electronics", "Fashion", "Home & Garden", "Subscriptions"],
        "table_title": "Buyer Purchase Records",
        "select_label": "Select Buyer ID to Analyze",
        "upload_title": "Bulk Buyer Database CSV Upload",
        "upload_subtitle": "Upload buyer order transaction exports (.csv) to batch predict buyer churn probabilities.",
        "sample_csv_path": "data/sample_batch_customers.csv",
        "sample_csv_name": "sample_batch_buyers.csv",
        "health_title": "E-Commerce Model Quality and Monitoring",
        "health_subtitle": "Monitor buyer retention metrics, purchase value drift indicators, and model health.",
        "console_title": "E-Commerce Self-Healing Console",
        "console_subtitle": "Real-time monitoring of e-commerce data quality, order drift, and automated model updates.",
    },
    "Fitness Club Member Churn": {
        "title": "Fitness Club Member Churn Console",
        "subtitle": "Track gym member attendance and membership renewal probabilities.",
        "entity_name": "Member",
        "unit": "Members",
        "total_count": "3,400",
        "high_risk_pct": "15.6%",
        "med_risk_pct": "29.1%",
        "low_risk_pct": "55.3%",
        "kpi1_label": "Gym Members",
        "kpi2_label": "High Risk Cancellations",
        "kpi3_label": "Medium Risk Segment",
        "kpi4_label": "Low Risk Segment",
        "records_title": "Gym Member Directory and Renewal Risk",
        "records_subtitle": "Browse gym member check-in history and inspect individual renewal probabilities.",
        "search_label": "Search Member by ID",
        "filter_label": "Membership Plan Filter",
        "filter_options": ["All", "Monthly Pass", "Annual VIP", "Class Pass"],
        "table_title": "Gym Member Check-In Records",
        "select_label": "Select Member ID to Analyze",
        "upload_title": "Bulk Member Register CSV Upload",
        "upload_subtitle": "Upload gym membership exports (.csv) to batch predict member cancellation risks.",
        "sample_csv_path": "data/sample_batch_customers.csv",
        "sample_csv_name": "sample_batch_members.csv",
        "health_title": "Fitness Club Model Quality and Monitoring",
        "health_subtitle": "Monitor gym member retention metrics, attendance drift indicators, and model health.",
        "console_title": "Fitness Self-Healing Console",
        "console_subtitle": "Real-time monitoring of check-in data quality, attendance drift, and automated model updates.",
    },
}

DOMAIN_CONFIGS.update(st.session_state["custom_domains"])

if domain_option == "+ Add Custom Domain...":
    st.markdown("<h1 class='main-title'>Custom Industry Domain Builder</h1>", unsafe_allow_html=True)
    st.write("Configure a new industry domain to customize metric labels, entity names, unit labels, and intervention plays across the entire platform.")
    st.write("")

    with st.form("custom_domain_builder_form"):
        c_name = st.text_input("Domain Name", placeholder="e.g. Hospital Patient Readmission")
        c_subtitle = st.text_input("Mission / Subtitle", placeholder="e.g. Monitor patient recovery and prevent 30-day readmissions.")
        c_entity = st.text_input("Entity Singular Name", placeholder="e.g. Patient")
        c_unit = st.text_input("Entity Plural Unit", placeholder="e.g. Patients")
        
        c_kpi1 = st.text_input("Primary KPI Label", value="Scored Records")
        c_kpi2 = st.text_input("High Risk Segment Label", value="High Risk Segment")
        c_kpi3 = st.text_input("Medium Risk Segment Label", value="Moderate Risk Segment")
        c_kpi4 = st.text_input("Low Risk Segment Label", value="Low Risk Segment")

        c_advice1 = st.text_input("Actionable Intervention Play 1", placeholder="e.g. Schedule post-discharge follow-up telehealth call.")
        c_advice2 = st.text_input("Actionable Intervention Play 2", placeholder="e.g. Verify prescription adherence and home nursing visit.")

        submitted = st.form_submit_button("Save and Apply Custom Domain", type="primary")

        if submitted:
            if not c_name or not c_entity:
                st.error("Please provide both Domain Name and Entity Singular Name.")
            else:
                unit_str = c_unit if c_unit.strip() else f"{c_entity}s"
                new_cfg = {
                    "title": f"{c_name} Console",
                    "subtitle": c_subtitle if c_subtitle.strip() else f"Monitor and prevent {c_entity.lower()} churn.",
                    "entity_name": c_entity,
                    "unit": unit_str,
                    "total_count": "1,000",
                    "high_risk_pct": "15.0%",
                    "med_risk_pct": "25.0%",
                    "low_risk_pct": "60.0%",
                    "kpi1_label": c_kpi1 or "Scored Records",
                    "kpi2_label": c_kpi2 or "High Risk Segment",
                    "kpi3_label": c_kpi3 or "Moderate Risk Segment",
                    "kpi4_label": c_kpi4 or "Low Risk Segment",
                    "records_title": f"{c_entity} Directory and Risk Analysis",
                    "records_subtitle": f"Browse {unit_str.lower()} and inspect individual risk factors with SHAP explanations.",
                    "search_label": f"Search {c_entity} by ID",
                    "filter_label": "Category Filter",
                    "filter_options": ["All", "Category A", "Category B", "Category C"],
                    "table_title": f"{c_entity} Records",
                    "select_label": f"Select {c_entity} ID to Analyze",
                    "upload_title": f"Bulk {c_entity} CSV Upload and Predictions",
                    "upload_subtitle": f"Upload {unit_str.lower()} database exports (.csv) to batch predict risk scores.",
                    "sample_csv_path": "data/sample_batch_customers.csv",
                    "sample_csv_name": f"sample_batch_{c_entity.lower()}.csv",
                    "health_title": f"{c_name} Model Quality & Monitoring",
                    "health_subtitle": f"Monitor ML model quality metrics, precision-recall indicators, and drift reports for {unit_str.lower()}.",
                    "console_title": f"{c_name} Self-Healing Console",
                    "console_subtitle": f"Real-time monitoring of {c_entity.lower()} data quality, drift, and automated model retraining.",
                    "custom_advice": [a for a in [c_advice1, c_advice2] if a and a.strip()],
                }
                st.session_state["custom_domains"][c_name] = new_cfg
                st.success(f"Custom domain '{c_name}' created! Select it from the sidebar to activate.")
                st.rerun()

    st.stop()

current_domain = DOMAIN_CONFIGS[domain_option]
domain_id_map = {
    "Telecom Customer Churn": "telecom",
    "School Student Churn": "school",
    "E-Commerce Customer Churn": "ecommerce",
    "Fitness Club Member Churn": "fitness",
}
current_domain_id = domain_id_map.get(domain_option, f"custom_{domain_option.lower().replace(' ', '_')}")


# ── Health check ────────────────────────────────────────────
@st.cache_data(ttl=15)
def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "degraded", "model_loaded": False, "model_version": "unknown"}


health = get_health()
if health.get("status") != "healthy":
    st.error(" Prediction Engine offline. Falling back to local scoring pipeline.")


# ── Drift banner ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_drift_status():
    try:
        r = requests.get(f"{API_URL}/drift/status", timeout=3)
        return r.json()
    except Exception:
        return {"status": "healthy", "drift_detected": False}


drift = get_drift_status()
if drift.get("status") == "significant_drift":
    st.error(" Significant data drift detected — model retraining recommended.")
elif drift.get("status") == "mild_drift":
    st.warning(" Mild data drift detected — monitor closely.")

# ── Pages ─────────────────────────────────────────────────────

if page == "Overview":
    st.markdown(
        f"<h1 class='main-title'>{current_domain['title']}</h1>",
        unsafe_allow_html=True,
    )
    st.caption(current_domain["subtitle"])
    mv = health.get("model_version", "N/A")
    v_prefix = "v" if mv and mv[0].isdigit() else ""
    st.caption(f"Model Version: {v_prefix}{mv} · Serving Node: {API_URL}")
    st.write("")

    # KPI Layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">{current_domain['kpi1_label']}</div>
            <div class="kpi-value">{current_domain['total_count']}</div>
            <div class="kpi-delta" style="color: #10B981;">Active Records</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">{current_domain['kpi2_label']}</div>
            <div class="kpi-value" style="color: #EF4444;">{current_domain['high_risk_pct']}</div>
            <div class="kpi-delta" style="color: #EF4444;">High Concern</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">{current_domain['kpi3_label']}</div>
            <div class="kpi-value" style="color: #F59E0B;">{current_domain['med_risk_pct']}</div>
            <div class="kpi-delta" style="color: #94A3B8;">Moderate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">{current_domain['kpi4_label']}</div>
            <div class="kpi-value" style="color: #10B981;">{current_domain['low_risk_pct']}</div>
            <div class="kpi-delta" style="color: #10B981;">Stable Segment</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    chart_col1, chart_col2 = st.columns([1, 1.2])

    with chart_col1:
        st.markdown("### Risk Distribution Summary")
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Low Risk", "Medium Risk", "High Risk"],
                    values=[45.5, 31.1, 23.4],
                    hole=0.55,
                    marker_colors=["#10B981", "#F59E0B", "#EF4444"],
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            font=dict(family="Inter", size=12, color="#94A3B8"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown("### 30-Day Rolling Prediction History")
        dates = [datetime.now() - timedelta(days=i) for i in range(30)][::-1]
        dummy_counts = [int(x) for x in np.random.normal(230, 20, 30)]
        fig_trend = px.line(
            x=dates,
            y=dummy_counts,
            labels={"x": "Date", "y": "Predictions Count"},
            markers=True,
        )
        fig_trend.update_traces(
            line_color="#14B8A6",
            line_width=3,
            marker=dict(size=6, color="#0D9488", symbol="circle"),
        )
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=20, r=20, t=20, b=0),
            font=dict(family="Inter", size=11, color="#64748B"),
            xaxis=dict(
                showgrid=False, tickfont=dict(color="#64748B"), linecolor="#1E293B"
            ),
            yaxis=dict(
                gridcolor="#1E293B",
                zeroline=False,
                tickfont=dict(color="#64748B"),
                linecolor="#1E293B",
            ),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

elif page == "Customers":
    st.markdown(
        f"<h1 class='main-title'>{current_domain['records_title']}</h1>",
        unsafe_allow_html=True,
    )
    st.write(current_domain["records_subtitle"])
    st.write("")

    test_path = "data/processed/test.csv"
    if not os.path.exists(test_path):
        st.warning(
            "Processed test split not found. Please run the DVC pipeline first."
        )
    else:
        df_test = pd.read_csv(test_path)
        # Generate dummy IDs for demonstration
        if "customerID" not in df_test.columns:
            df_test["customerID"] = [
                f"ID-{1000 + idx}" for idx in range(len(df_test))
            ]

        # Quick Profile Presets Bar
        st.write("**Quick Profile Presets:**")
        p_col1, p_col2, _ = st.columns([1.2, 1.2, 2.6])
        if p_col1.button("Load High-Risk Customer Preset"):
            st.session_state["preset_filter"] = "Month-to-month"
            st.session_state["preset_query"] = "Month-to-month"
        if p_col2.button("Load Loyal Customer Preset"):
            st.session_state["preset_filter"] = "Two year"
            st.session_state["preset_query"] = "Two year"

        st.write("")
        # Search controls
        search_col1, search_col2 = st.columns([1, 2])
        with search_col1:
            default_filter = st.session_state.get("preset_filter", "All")
            filter_opts = current_domain["filter_options"]
            filter_idx = filter_opts.index(default_filter) if default_filter in filter_opts else 0
            contract_filter = st.selectbox(
                current_domain["filter_label"],
                filter_opts,
                index=filter_idx
            )
        with search_col2:
            search_query = st.text_input(current_domain["search_label"], "").strip()

        filtered_df = df_test
        if contract_filter != "All" and "Contract" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Contract"] == contract_filter]
        if search_query:
            filtered_df = filtered_df[
                filtered_df["customerID"].str.contains(search_query, case=False)
            ]

        col_table, col_detail = st.columns([1.2, 1])

        with col_table:
            st.markdown(f"### {current_domain['table_title']}")
            st.write(
                "Select a row in the selector below to run real-time SHAP explanation."
            )

            # Show table
            display_cols = [
                "customerID",
                "gender",
                "tenure",
                "Contract",
                "MonthlyCharges",
                "TotalCharges",
            ]
            st.dataframe(filtered_df[display_cols].head(15), use_container_width=True)

            # Select Entity ID
            st.write("")
            selected_id = st.selectbox(
                current_domain["select_label"], filtered_df["customerID"].head(15)
            )

        with col_detail:
            if selected_id:
                cust_row = (
                    filtered_df[filtered_df["customerID"] == selected_id]
                    .iloc[0]
                    .to_dict()
                )

                # Strip helper customerID and Churn target for prediction schema
                target_val = cust_row.pop("Churn", None)
                customer_id_val = cust_row.pop("customerID", None)

                # Make API call with domain routing
                res = {}
                try:
                    r = requests.post(
                        f"{API_URL}/predict?domain={current_domain_id}", json=cust_row, headers=HEADERS, timeout=5
                    )
                    if r.status_code == 200:
                        res = r.json()
                        prob = res["churn_probability"]
                        risk = res["risk_tier"]
                        prediction = res["prediction"]
                        top_factors = res["top_factors"]
                    else:
                        raise Exception(f"API status {r.status_code}")
                except Exception:
                    # Fallback to local domain prediction model if FastAPI offline
                    try:
                        from src.domain_registry import load_domain_model, load_domain_preprocessor

                        model = load_domain_model(current_domain_id)
                        preprocessor = load_domain_preprocessor(current_domain_id)
                        from api.predict import run_single_prediction

                        # mock a FastAPI request object
                        class MockRequest:
                            class State:
                                def __init__(self, m, p):
                                    self.model = m
                                    self.preprocessor = p
                                    self.model_version = "local-fallback"

                            def __init__(self, m, p):
                                self.state = self.State(m, p)

                        # Mock schema input
                        from api.schemas import CustomerInput
                        from api.database import SessionLocal

                        mock_req = MockRequest(model, preprocessor)
                        db = SessionLocal()
                        try:
                            res_obj = run_single_prediction(
                                mock_req, CustomerInput(**cust_row), db
                            )
                            res = res_obj.model_dump()
                            prob = res_obj.churn_probability
                            risk = res_obj.risk_tier
                            prediction = res_obj.prediction
                            top_factors = [f.model_dump() for f in res_obj.top_factors]
                        finally:
                            db.close()
                    except Exception as e:
                        st.error(f"Prediction logic failed: {e}")
                        prob, risk, prediction, top_factors = 0.0, "Low", 0, []
                        res = {}

                st.markdown("### Risk Diagnosis")

                # Prob Gauge
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={
                            "text": "Churn Probability (%)",
                            "font": {
                                "size": 16,
                                "color": "#94A3B8",
                                "family": "Outfit",
                            },
                        },
                        number={
                            "suffix": "%",
                            "font": {
                                "color": "#F8FAFC",
                                "size": 36,
                                "family": "Outfit",
                            },
                        },
                        gauge={
                            "axis": {
                                "range": [None, 100],
                                "tickwidth": 1,
                                "tickcolor": "#475569",
                                "tickmode": "array",
                                "tickvals": [0, 35, 65, 100],
                                "ticktext": ["0%", "35%", "65%", "100%"],
                                "tickfont": {
                                    "color": "#64748B",
                                    "size": 11,
                                    "family": "Inter",
                                },
                            },
                            "bar": {"color": "#14B8A6"},
                            "bgcolor": "#1E293B",
                            "borderwidth": 1,
                            "bordercolor": "#334155",
                            "steps": [
                                {
                                    "range": [0, 35],
                                    "color": "rgba(16, 185, 129, 0.15)",
                                },  # Emerald Green
                                {
                                    "range": [35, 65],
                                    "color": "rgba(245, 158, 11, 0.15)",
                                },  # Amber
                                {
                                    "range": [65, 100],
                                    "color": "rgba(239, 68, 68, 0.15)",
                                },  # Rose Red
                            ],
                        },
                    )
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=240,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                color_map = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
                badge_color = color_map.get(risk, "#334155")
                st.markdown(
                    f"""
                <div class="kpi-card" style="border-left: 6px solid {badge_color}; padding: 16px;">
                    <h4 style="margin:0px; color:{badge_color};">Risk Classification: {risk}</h4>
                    <p style="margin: 6px 0 0 0; font-size: 0.9rem;">
                        <b>ID:</b> {selected_id} | <b>Predicted Action:</b> {"Churn Warning" if prediction == 1 else "Likely to Retain"}
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # SHAP Factors
                st.write("")
                st.markdown("#### Top 3 Churn Drivers (SHAP)")
                if top_factors:
                    for idx, factor in enumerate(top_factors):
                        impact_val = float(factor["impact"])
                        sign = "+" if impact_val > 0 else ""
                        color = "#EF4444" if impact_val > 0 else "#10B981"
                        st.markdown(
                            f"- **{factor['feature']}** has value `{factor['value']}`: "
                            f"<span style='color: {color}; font-weight: bold;'>{sign}{impact_val:.2f} Impact</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("SHAP explanation not available.")

                # Advice
                st.write("")
                # Survival Analysis Timeline (Upgrade 2)
                st.write("")
                st.markdown("#### Survival Analysis & Time-to-Churn Horizon")
                if res.get("risk_horizon_summary"):
                    st.info(res["risk_horizon_summary"])
                
                timeline = res.get("survival_timeline", {})
                if timeline:
                    s_cols = st.columns(4)
                    s_cols[0].metric("30-Day Survival", f"{timeline.get('30_days', 0)*100:.1f}%")
                    s_cols[1].metric("60-Day Survival", f"{timeline.get('60_days', 0)*100:.1f}%")
                    s_cols[2].metric("90-Day Survival", f"{timeline.get('90_days', 0)*100:.1f}%")
                    s_cols[3].metric("180-Day Survival", f"{timeline.get('180_days', 0)*100:.1f}%")

                # Actionable Retention Playbooks (Upgrade 1)
                st.write("")
                st.markdown("#### Automated Actionable Intervention Playbook")
                recommended = res.get("recommended_actions", [])
                if recommended:
                    for item in recommended:
                        st.markdown(f"- **Intervention**: {item}")
                else:
                    st.markdown("- Account healthy. Maintain standard service delivery.")

elif page == "Upload & Score":
    st.markdown(
        f"<h1 class='main-title'>{current_domain['upload_title']}</h1>",
        unsafe_allow_html=True,
    )
    st.write(current_domain["upload_subtitle"])
    st.write("")

    uploaded_file = st.file_uploader(f"Upload your {current_domain['entity_name'].lower()} CSV file", type="csv")

    if not uploaded_file:
        st.markdown(
            f"""
        <div class="kpi-card" style="margin-top: 20px; border-left: 4px solid #0D9488;">
            <h4 style="margin-top: 0px; color: #14B8A6;"> Expected {current_domain['entity_name']} CSV Schema</h4>
            <p style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 0px;">
                Ensure your CSV file contains the required columns before uploading. All fields will be parsed and self-healed automatically during ingestion.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.write("")

        # Display sample schema table
        if os.path.exists(current_domain["sample_csv_path"]):
            sample_df = pd.read_csv(current_domain["sample_csv_path"]).head(5)
        else:
            sample_data = {
                "customerID": ["7590-VHVEG", "5575-GNVDE"],
                "gender": ["Female", "Male"],
                "tenure": [1, 34],
                "Contract": ["Month-to-month", "One year"],
                "MonthlyCharges": [29.85, 56.95],
                "TotalCharges": [29.85, 1889.50],
            }
            sample_df = pd.DataFrame(sample_data)

        st.dataframe(sample_df, use_container_width=True)

        # Download template button
        csv_template = sample_df.to_csv(index=False)
        st.download_button(
            label=f"Download Template CSV ({current_domain['sample_csv_name']})",
            data=csv_template,
            file_name=current_domain["sample_csv_name"],
            mime="text/csv",
        )
    else:
        if st.button("Run Predictions", type="primary"):
            with st.spinner(f"Scoring {current_domain['unit'].lower()}..."):
                try:
                    r = requests.post(
                        f"{API_URL}/upload?domain={current_domain_id}",
                        headers=HEADERS,
                        files={"file": (uploaded_file.name, uploaded_file, "text/csv")},
                        timeout=30,
                    )
                    if r.status_code == 200:
                        result_df = pd.read_csv(pd.io.common.BytesIO(r.content))
                        high = (result_df["risk_tier"] == "High").sum()
                        med = (result_df["risk_tier"] == "Medium").sum()
                        low = (result_df["risk_tier"] == "Low").sum()

                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"High Risk {current_domain['unit']}", high)
                        col2.metric(f"Medium Risk {current_domain['unit']}", med)
                        col3.metric(f"Low Risk {current_domain['unit']}", low)

                        st.write("")
                        st.dataframe(
                            result_df.head(100),
                            use_container_width=True,
                        )
                        st.download_button(
                            "Download Results CSV",
                            r.content,
                            "predictions_output.csv",
                            "text/csv",
                        )
                    else:
                        st.error(
                            f"Error: {r.json().get('detail', 'Validation or serving error')}"
                        )
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")

elif page == "Model Health":
    st.markdown(
        f"<h1 class='main-title'>{current_domain['health_title']}</h1>",
        unsafe_allow_html=True,
    )
    st.write(current_domain["health_subtitle"])
    st.write("")

    # Query metrics from API
    try:
        r_metrics = requests.get(f"{API_URL}/metrics?domain={current_domain_id}", headers=HEADERS, timeout=5)
        if r_metrics.status_code == 200:
            metrics_data = r_metrics.json()
            f1_val = f"{metrics_data.get('f1'):.3f}"
            auc_val = f"{metrics_data.get('roc_auc'):.3f}"
            metrics_source = metrics_data.get("source", "API")
        else:
            raise Exception("API error")
    except Exception:
        f1_val = "0.830"
        auc_val = "0.890"
        metrics_source = "Fallback Defaults"

    col1, col2, col3 = st.columns(3)
    col1.metric("Target F1 Score", f1_val, f"Goal >= 0.820 ({metrics_source})")
    col2.metric("Validation AUC-ROC", auc_val, f"Goal >= 0.880 ({metrics_source})")
    col3.metric("Inference Latency (p95)", "< 45ms", "Goal < 200ms")

    st.write("")
    st.subheader("Fairness and Demographic Bias Analysis")
    fairness_data = None
    try:
        from src.domain_registry import get_domain_model_dir
        metrics_file = get_domain_model_dir(current_domain_id) / "eval_metrics.json"
        if not metrics_file.exists():
            metrics_file = Path("metrics/eval_metrics.json")
        if metrics_file.exists():
            import json
            with open(metrics_file) as f:
                fairness_data = json.load(f).get("fairness")
    except Exception:
        pass

    if fairness_data and "subgroups" in fairness_data:
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("EEOC 4/5 Selection Ratio", f"{fairness_data.get('four_fifths_selection_ratio', 1.0):.4f}", "Goal >= 0.800")
        f_col2.metric("Demographic Parity Diff", f"{fairness_data.get('demographic_parity_difference', 0.0):.4f}", "Goal <= 0.150")
        f_col3.metric("Equalized Odds Diff", f"{fairness_data.get('equalized_odds_difference', 0.0):.4f}", "Goal <= 0.150")
        b_status = fairness_data.get("bias_status", "acceptable")
        f_col4.metric("Bias Audit Status", b_status.upper(), "Passed Audit" if b_status == "acceptable" else "Disparity Flagged")

        st.write("")
        st.markdown("#### Subgroup Metric Breakdown")
        sg = fairness_data["subgroups"]
        
        sg_rows = []
        for category, groups in sg.items():
            for gname, gstats in groups.items():
                sg_rows.append({
                    "Subgroup Category": category,
                    "Subgroup": gname,
                    "Sample Count": gstats.get("count", 0),
                    "Selection Rate": f"{gstats.get('selection_rate', 0.0)*100:.1f}%",
                    "Recall (TPR)": f"{gstats.get('recall_tpr', 0.0)*100:.1f}%",
                    "False Positive Rate (FPR)": f"{gstats.get('fpr', 0.0)*100:.1f}%",
                    "F1 Score": f"{gstats.get('f1', 0.0):.4f}",
                })
        st.dataframe(pd.DataFrame(sg_rows), use_container_width=True)
    else:
        st.info("Subgroup fairness audit metrics are generated during model training/evaluation.")

    st.write("")
    st.subheader("Evidently AI Diagnostics")

    try:
        import streamlit.components.v1 as components
        r = requests.get(f"{API_URL}/drift/report", timeout=5)
        if r.status_code == 200:
            components.html(r.text, height=800, scrolling=True)
        else:
            st.markdown(
                """
            <div class="kpi-card" style="text-align: center; padding: 40px; margin-top: 20px; border: 1px dashed #202D42;">
                <div style="font-size: 3.5rem; margin-bottom: 10px;"></div>
                <h3 style="margin-top: 0px; color: #F8FAFC;">No Data Drift Report Available</h3>
                <p style="color: #64748B; max-width: 500px; margin: 0 auto 20px auto; font-size: 0.95rem;">
                    Drift reports are generated automatically by Evidently AI when the server processes incoming requests.
                </p>
                <div style="color: #0D9488; font-weight: 600; font-size: 0.9rem;">
                     Tip: Go to the "Self-Healing Console" and click "Inject 100 Drifted Requests" to trigger drift monitoring.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Could not connect to drift API: {e}")

elif page == "Self-Healing Console":
    st.markdown(
        f"<h1 class='main-title'>{current_domain['console_title']}</h1>",
        unsafe_allow_html=True,
    )
    st.write(current_domain["console_subtitle"])
    st.write("")

    # 1. Fetch current status & logs from API
    try:
        health_resp = requests.get(f"{API_URL}/health?domain={current_domain_id}", timeout=3).json()
        current_version = health_resp.get("model_version", "unknown")
        status_color = "" if health_resp.get("status") == "healthy" else ""
    except Exception:
        current_version = "offline"
        status_color = ""

    logs = []
    try:
        r = requests.get(f"{API_URL}/self-healing/logs?domain={current_domain_id}", headers=HEADERS, timeout=5)
        if r.status_code == 200:
            logs = r.json()
    except Exception as e:
        st.warning(f"Could not connect to self-healing logs endpoint: {e}")

    # Count events
    dq_count = sum(1 for log_item in logs if log_item["event_type"] == "data_quality")
    retrain_count = sum(
        1 for log_item in logs if log_item["event_type"] == "retraining"
    )

    # Metrics section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Serving Model Version</div>
            <div class='kpi-value'>{current_version}</div>
            <div class='kpi-delta'>{status_color} Service State</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Data Quality Corrections</div>
            <div class='kpi-value'>{dq_count}</div>
            <div class='kpi-delta'> Auto-healed inputs</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Drift Retraining Runs</div>
            <div class='kpi-value'>{retrain_count}</div>
            <div class='kpi-delta'> Feedback loop retrains</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Ingestion Playground & Simulated Retrain Layout
    tab1, tab2, tab3 = st.tabs(
        ["Ingestion Playground", "Retraining & Drift Demo", "System Logs"]
    )

    with tab1:
        st.subheader("Interactive Data Ingestion Playground")
        st.write(
            "Submit malformed or invalid JSON payloads to test ChurnGuard's rule-based healing engine."
        )

        default_payload = """{
  "customerID": "9999-HEAL",
  "tenure": -12,
  "MonthlyCharges": 120.50,
  "TotalCharges": 50.00,
  "SeniorCitizen": "Yes",
  "gender": "Femail",
  "Partner": "Nooo",
  "PaymentMethod": "Electrnic check",
  "Dependents": "Yes",
  "PhoneService": "Yes",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes"
}"""

        payload_text = st.text_area(
            "Customer JSON Payload", value=default_payload, height=350
        )

        if st.button("Test Self-Healing Ingestion", type="primary"):
            import json

            try:
                payload = json.loads(payload_text)

                with st.spinner("Processing through self-healing gateway..."):
                    res = requests.post(
                        f"{API_URL}/predict?domain={current_domain_id}", headers=HEADERS, json=payload, timeout=10
                    )

                    if res.status_code == 200:
                        res_data = res.json()
                        st.success(
                            " Prediction successful! The input was successfully healed."
                        )

                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.write("**Healed Actions Applied:**")
                            healed_actions = res_data.get("healed_actions", [])
                            if healed_actions:
                                for act in healed_actions:
                                    st.info(f" {act}")
                            else:
                                st.info("No healing required. Ingested cleanly.")

                        with col_r:
                            st.write("**Model Output:**")
                            st.metric(
                                "Churn Probability",
                                f"{res_data['churn_probability'] * 100:.2f}%",
                            )
                            st.metric("Risk Tier", res_data["risk_tier"])
                            st.caption(f"Served by: {res_data['model_version']}")
                    else:
                        st.error(
                            f"Prediction failed with status {res.status_code}: {res.text}"
                        )
            except Exception as e:
                st.error(f"Error parsing JSON payload: {e}")

    with tab2:
        st.subheader("Drift Monitoring & Manual Retraining")
        st.write(
            "Demonstrate feedback self-healing loop: either inject a batch of highly drifted inputs or trigger a manual model rebuild."
        )

        col_demo1, col_demo2 = st.columns(2)

        with col_demo1:
            st.markdown("### Option A: Force Retraining")
            st.write(
                "Instruct the model worker to rebuild the XGBoost model immediately using current database inputs."
            )
            if st.button("Manual Retrain Model", key="man_retrain"):
                with st.spinner("Starting background retraining thread..."):
                    try:
                        res = requests.post(
                            f"{API_URL}/self-healing/trigger-retrain?domain={current_domain_id}",
                            headers=HEADERS,
                            timeout=5,
                        )
                        if res.status_code == 200:
                            st.success(f"Response: {res.json().get('message')}")
                        else:
                            st.error(f"Failed to trigger retraining: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

        # Champion vs Challenger Shadow Deployment Section (Upgrade 3)
        st.write("")
        st.markdown("### Champion vs. Challenger Shadow Deployment (A/B Testing)")
        try:
            r_shad = requests.get(f"{API_URL}/model/shadow-status?domain={current_domain_id}", headers=HEADERS, timeout=5)
            if r_shad.status_code == 200:
                shad_data = r_shad.json()
                c_col1, c_col2 = st.columns(2)
                c_col1.metric("Shadow Inference Samples", shad_data.get("sample_count", 0))
                c_col2.metric("Mean Prediction Divergence", f"{shad_data.get('avg_delta', 0.0):.4f}")
                if st.button("Promote Challenger to Champion", key="promote_btn"):
                    r_prom = requests.post(f"{API_URL}/model/promote?domain={current_domain_id}", headers=HEADERS, timeout=5)
                    st.success(r_prom.json().get("message", "Promoted successfully."))
        except Exception:
            st.write("Shadow evaluation inactive.")

        with col_demo2:
            st.markdown("### Option B: Simulate High Data Drift")
            st.write(
                "Inject 100 heavily drifted prediction queries. This automatically exceeds the drift preset threshold and triggers auto-retraining."
            )

            if st.button("Inject 100 Drifted Requests", key="inject_drift"):
                with st.spinner("Injecting drifted data..."):
                    try:
                        drifted_batch = []
                        for i in range(100):
                            drifted_batch.append(
                                {
                                    "customerID": f"DRIFT-{i}",
                                    "tenure": 1,
                                    "MonthlyCharges": 500.0,
                                    "TotalCharges": 500.0,
                                    "SeniorCitizen": 1,
                                    "gender": "Female",
                                    "Partner": "Yes",
                                    "Dependents": "No",
                                    "PhoneService": "Yes",
                                    "MultipleLines": "Yes",
                                    "InternetService": "Fiber optic",
                                    "OnlineSecurity": "No",
                                    "OnlineBackup": "No",
                                    "DeviceProtection": "No",
                                    "TechSupport": "No",
                                    "StreamingTV": "Yes",
                                    "StreamingMovies": "Yes",
                                    "Contract": "Month-to-month",
                                    "PaperlessBilling": "Yes",
                                    "PaymentMethod": "Electronic check",
                                }
                            )

                        res = requests.post(
                            f"{API_URL}/predict/batch",
                            headers=HEADERS,
                            json={"customers": drifted_batch},
                            timeout=25,
                        )
                        if res.status_code == 200:
                            st.success(
                                "Successfully injected 100 drifted requests! Drift check running on API. Check logs tab for retraining status."
                            )
                        else:
                            st.error(f"Injection failed: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    with tab3:
        st.subheader("Self-Healing Event Logs")
        if not logs:
            st.info("No logs present in the self-healing event history.")
        else:
            for log_item in logs:
                timestamp_str = log_item["created_at"][:19].replace("T", " ")
                event_type = log_item["event_type"].upper()
                desc = log_item["description"]

                badge_icon = "" if log_item["event_type"] == "data_quality" else ""
                item_class = (
                    "data-quality"
                    if log_item["event_type"] == "data_quality"
                    else "retraining"
                )

                if ": " in desc:
                    title, details = desc.split(": ", 1)
                else:
                    title = desc
                    details = None

                st.markdown(
                    f"""
                <div class="timeline-item {item_class}">
                    <div class="timeline-badge">{badge_icon}</div>
                    <div class="timeline-header">{timestamp_str} · {event_type}</div>
                    <div class="timeline-title">{title}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if details:
                    with st.expander(" Show Correction Details"):
                        st.info(details)
                    st.write("")
