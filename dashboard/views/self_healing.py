"""
Self-healing event log, retraining, and champion/challenger control.
"""

import json

import streamlit as st

from dashboard import api_client


def render(current_domain, current_domain_id, health, drift):
    st.markdown(
        f"<h1 class='main-title'>{current_domain['console_title']}</h1>",
        unsafe_allow_html=True,
    )
    st.write(current_domain["console_subtitle"])
    st.write("")

    # 1. Fetch current status & logs from API
    try:
        health_resp = api_client.get_health(domain=current_domain_id)
        current_version = health_resp.get("model_version", "unknown")
        status_color = "" if health_resp.get("status") == "healthy" else ""
    except Exception:
        current_version = "offline"
        status_color = ""

    logs = []
    try:
        logs = api_client.get_self_healing_logs(domain=current_domain_id)
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
            try:
                payload = json.loads(payload_text)

                with st.spinner("Processing through self-healing gateway..."):
                    res = api_client.predict(payload, domain=current_domain_id)

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
                        res = api_client.trigger_retrain(domain=current_domain_id)
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
            shad_data = api_client.get_shadow_status(domain=current_domain_id)
            if shad_data:
                c_col1, c_col2 = st.columns(2)
                c_col1.metric(
                    "Shadow Inference Samples", shad_data.get("sample_count", 0)
                )
                c_col2.metric(
                    "Mean Prediction Divergence",
                    f"{shad_data.get('avg_delta', 0.0):.4f}",
                )
                if st.button("Promote Challenger to Champion", key="promote_btn"):
                    r_prom = api_client.promote_model(domain=current_domain_id)
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

                        res = api_client.predict_batch(
                            drifted_batch, domain=current_domain_id
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
