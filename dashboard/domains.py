"""
Domain presentation config and the custom-domain builder.

These control labels and copy only. The mapping from a display name to the
API's domain key is NOT duplicated here — it comes from
src.domain_registry.sanitize_domain_id, which is what the API itself uses.
The dashboard used to keep its own copy that derived custom keys differently
(`custom_{name}` vs the sanitizer), so a domain whose name contained
punctuation resolved to a different key in the UI than in the API.
"""

import streamlit as st

from dashboard import api_client
from src.domain_registry import sanitize_domain_id

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
        "filter_options": [
            "All",
            "Grade 1-5 (Primary)",
            "Grade 6-8 (Middle)",
            "Grade 9-12 (High School)",
        ],
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
        "filter_options": [
            "All",
            "Electronics",
            "Fashion",
            "Home & Garden",
            "Subscriptions",
        ],
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


def render_builder() -> None:
    """
    Render the custom-domain builder form.

    Saves the new config into session state and bootstraps the domain in the
    API so its artifacts and baseline exist before anyone selects it.
    """
    st.markdown(
        "<h1 class='main-title'>Custom Industry Domain Builder</h1>",
        unsafe_allow_html=True,
    )
    st.write(
        "Configure a new industry domain to customize metric labels, entity names, unit labels, and intervention plays across the entire platform."
    )
    st.write("")

    with st.form("custom_domain_builder_form"):
        c_name = st.text_input(
            "Domain Name", placeholder="e.g. Hospital Patient Readmission"
        )
        c_subtitle = st.text_input(
            "Mission / Subtitle",
            placeholder="e.g. Monitor patient recovery and prevent 30-day readmissions.",
        )
        c_entity = st.text_input("Entity Singular Name", placeholder="e.g. Patient")
        c_unit = st.text_input("Entity Plural Unit", placeholder="e.g. Patients")

        c_kpi1 = st.text_input("Primary KPI Label", value="Scored Records")
        c_kpi2 = st.text_input("High Risk Segment Label", value="High Risk Segment")
        c_kpi3 = st.text_input(
            "Medium Risk Segment Label", value="Moderate Risk Segment"
        )
        c_kpi4 = st.text_input("Low Risk Segment Label", value="Low Risk Segment")

        c_advice1 = st.text_input(
            "Actionable Intervention Play 1",
            placeholder="e.g. Schedule post-discharge follow-up telehealth call.",
        )
        c_advice2 = st.text_input(
            "Actionable Intervention Play 2",
            placeholder="e.g. Verify prescription adherence and home nursing visit.",
        )

        submitted = st.form_submit_button(
            "Save and Apply Custom Domain", type="primary"
        )

        if submitted:
            if not c_name or not c_entity:
                st.error("Please provide both Domain Name and Entity Singular Name.")
            else:
                unit_str = c_unit if c_unit.strip() else f"{c_entity}s"
                new_cfg = {
                    "title": f"{c_name} Console",
                    "subtitle": (
                        c_subtitle
                        if c_subtitle.strip()
                        else f"Monitor and prevent {c_entity.lower()} churn."
                    ),
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
                    "custom_advice": [
                        a for a in [c_advice1, c_advice2] if a and a.strip()
                    ],
                }
                st.session_state["custom_domains"][c_name] = new_cfg

                # Create the domain's artifacts and baseline in the API too;
                # otherwise it exists only in this browser session and the
                # first prediction against it fails.
                try:
                    resp = api_client.bootstrap_domain(c_name)
                    if resp.status_code != 200:
                        st.warning(
                            f"Domain saved locally, but the API rejected the "
                            f"bootstrap ({resp.status_code}). Scoring may fail "
                            f"until an administrator creates it."
                        )
                except Exception as exc:
                    st.warning(
                        f"Domain saved locally, but the API is unreachable "
                        f"({exc}). Scoring may fail until it is created."
                    )

                st.success(
                    f"Custom domain '{c_name}' created! Select it from the sidebar to activate."
                )
                st.rerun()

    st.stop()


def domain_key(display_name: str) -> str:
    """Resolve a display name to the key the API uses."""
    return sanitize_domain_id(display_name)


def all_configs() -> dict:
    """Built-in configs plus any custom domains created this session."""
    configs = dict(DOMAIN_CONFIGS)
    configs.update(st.session_state.get("custom_domains", {}))
    return configs
