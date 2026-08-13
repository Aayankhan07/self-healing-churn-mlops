"""
Record directory: search, filter, and per-record SHAP explanations.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client


def render(current_domain, current_domain_id, health, drift):
    st.markdown(
        f"<h1 class='main-title'>{current_domain['records_title']}</h1>",
        unsafe_allow_html=True,
    )
    st.write(current_domain["records_subtitle"])
    st.write("")

    test_path = "data/processed/test.csv"
    if not os.path.exists(test_path):
        st.warning("Processed test split not found. Please run the DVC pipeline first.")
    else:
        df_test = pd.read_csv(test_path)
        # Generate dummy IDs for demonstration
        if "customerID" not in df_test.columns:
            df_test["customerID"] = [f"ID-{1000 + idx}" for idx in range(len(df_test))]

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
            filter_idx = (
                filter_opts.index(default_filter)
                if default_filter in filter_opts
                else 0
            )
            contract_filter = st.selectbox(
                current_domain["filter_label"], filter_opts, index=filter_idx
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

                # Strip the label and the identifier: neither is a feature, and
                # sending the label would leak the answer into the request.
                cust_row.pop("Churn", None)
                cust_row.pop("customerID", None)

                # Make API call with domain routing
                res = {}
                try:
                    r = api_client.predict(cust_row, domain=current_domain_id)
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
                        from src.domain_registry import (
                            load_domain_model,
                            load_domain_preprocessor,
                        )

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
                    s_cols[0].metric(
                        "30-Day Survival", f"{timeline.get('30_days', 0)*100:.1f}%"
                    )
                    s_cols[1].metric(
                        "60-Day Survival", f"{timeline.get('60_days', 0)*100:.1f}%"
                    )
                    s_cols[2].metric(
                        "90-Day Survival", f"{timeline.get('90_days', 0)*100:.1f}%"
                    )
                    s_cols[3].metric(
                        "180-Day Survival", f"{timeline.get('180_days', 0)*100:.1f}%"
                    )

                # Actionable Retention Playbooks (Upgrade 1)
                st.write("")
                st.markdown("#### Automated Actionable Intervention Playbook")
                recommended = res.get("recommended_actions", [])
                if recommended:
                    for item in recommended:
                        st.markdown(f"- **Intervention**: {item}")
                else:
                    st.markdown(
                        "- Account healthy. Maintain standard service delivery."
                    )
