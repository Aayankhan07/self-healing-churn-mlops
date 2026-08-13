"""
Streamlit executive dashboard.

Owns navigation and the shared context every page needs; the pages themselves
live in dashboard/pages/. Talks to the API only through dashboard.api_client —
no direct DB access.
"""

import streamlit as st

from dashboard import api_client, domains, styles
from dashboard.views import customers, model_health, overview
from dashboard.views import self_healing, upload

st.set_page_config(
    page_title="ChurnGuard — Customer Churn Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

styles.apply()

# ── Sidebar Navigation ──────────────────────────────────────
st.sidebar.markdown(
    "<h2 style='text-align: center; margin-bottom: 20px;'>ChurnGuard</h2>",
    unsafe_allow_html=True,
)

if "custom_domains" not in st.session_state:
    st.session_state["custom_domains"] = {}

ADD_DOMAIN_OPTION = "+ Add Custom Domain..."

domain_options = (
    [
        "Telecom Customer Churn",
        "School Student Churn",
        "E-Commerce Customer Churn",
        "Fitness Club Member Churn",
    ]
    + list(st.session_state["custom_domains"].keys())
    + [ADD_DOMAIN_OPTION]
)

domain_option = st.sidebar.selectbox("Industry Domain", domain_options)

PAGES = {
    "Overview": overview,
    "Customers": customers,
    "Upload & Score": upload,
    "Model Health": model_health,
    "Self-Healing Console": self_healing,
}

page = st.sidebar.radio("Navigation", list(PAGES))

# The builder takes over the whole view and stops the script.
if domain_option == ADD_DOMAIN_OPTION:
    domains.render_builder()

current_domain = domains.all_configs()[domain_option]
current_domain_id = domains.domain_key(domain_option)

# ── Service banners ─────────────────────────────────────────
health = api_client.get_health(domain=current_domain_id)
if health.get("status") != "healthy":
    st.error("Prediction Engine offline. Falling back to local scoring pipeline.")

if health.get("demo_fixture"):
    st.warning(
        f"'{domain_option}' is a demo fixture: it serves artifacts copied from "
        f"the Telecom domain rather than a model trained on its own data. "
        f"Scores are illustrative."
    )

drift = api_client.get_drift_status(domain=current_domain_id)
if drift.get("status") == "significant_drift":
    st.error("Significant data drift detected — model retraining recommended.")
elif drift.get("status") == "mild_drift":
    st.warning("Mild data drift detected — monitor closely.")

# ── Render the selected page ────────────────────────────────
PAGES[page].render(current_domain, current_domain_id, health, drift)
