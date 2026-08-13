"""
Bulk CSV upload and batch scoring.
"""

import os

import pandas as pd
import streamlit as st

from dashboard import api_client


def render(current_domain, current_domain_id, health, drift):
    st.markdown(
        f"<h1 class='main-title'>{current_domain['upload_title']}</h1>",
        unsafe_allow_html=True,
    )
    st.write(current_domain["upload_subtitle"])
    st.write("")

    uploaded_file = st.file_uploader(
        f"Upload your {current_domain['entity_name'].lower()} CSV file", type="csv"
    )

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
                    r = api_client.upload_csv(
                        uploaded_file.name, uploaded_file, domain=current_domain_id
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
