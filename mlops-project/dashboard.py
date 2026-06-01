import os
import json
import pandas as pd
import numpy as np
import httpx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

# Import local predictor as fallback if FastAPI is not active
from src.predict import ChurnPredictor

# Streamlit Page Setup
st.set_page_config(
    page_title="Telecom Customer Churn - MLOps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Harmonious Dark & Sleek Design
st.markdown("""
<style>
    .main {
        background-color: #0F111A;
        color: #E2E8F0;
    }
    .stApp {
        background-color: #0F111A;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #F8FAFC;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stButton>button {
        background-image: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Helper to fetch metrics
def load_metrics():
    metrics_path = "data/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

# Predict logic supporting API calls or local fallback
def get_prediction(payload: dict):
    api_url = "http://localhost:8000/predict"
    try:
        # Try to call FastAPI deployment service
        with httpx.Client(timeout=3.0) as client:
            response = client.post(api_url, json=payload)
            if response.status_code == 200:
                return response.json(), "FastAPI Serving Endpoint"
    except Exception:
        pass
    
    # Fallback to local prediction
    try:
        model_path = "data/model.pkl"
        local_svc = ChurnPredictor(model_path=model_path)
        results = local_svc.predict([payload])
        return results[0], "Local Pipeline Fallback (FastAPI offline)"
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None, None

# Sidebar / Header
st.title("📊 Telecom Customer Churn Prediction & Monitoring")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Real-Time Inference", "📈 Model Quality & Metrics", "🔬 Production Monitoring & Drift"])

with tab1:
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("👤 Enter Customer Attributes")
        st.write("Provide subscription specifics to calculate churn risk.")
        
        # User input controls matching model inputs
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12, step=1)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=15.0, max_value=130.0, value=70.0, step=0.5)
        
        # Approximate logical Total Charges
        approx_total = tenure * monthly_charges
        total_charges = st.number_input("Total Charges ($)", min_value=15.0, max_value=10000.0, value=approx_total, step=50.0)
        
        contract = st.selectbox("Contract Term", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        
        if internet_service != "No":
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No"])
        else:
            tech_support = "No internet service"
            online_security = "No internet service"
            
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        payload = {
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
            "Contract": contract,
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "PaperlessBilling": paperless_billing
        }
        
        st.write("")
        submit = st.button("🔮 Analyze Attrition Risk")

    with col_output:
        st.subheader("💡 Attrition Prediction Output")
        
        if submit:
            res, source = get_prediction(payload)
            if res:
                prob = res["churn_probability"]
                risk = res["churn_risk_level"]
                warning = res["attrition_warning"]
                
                # Gauge representation of Churn risk
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability (%)", 'font': {'size': 20, 'color': '#E2E8F0'}},
                    number = {'suffix': "%", 'font': {'color': '#E2E8F0', 'size': 36}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': "#6366F1"},
                        'bgcolor': "#1E293B",
                        'borderwidth': 2,
                        'bordercolor': "#334155",
                        'steps': [
                            {'range': [0, 40], 'color': '#10B981'},  # Green
                            {'range': [40, 70], 'color': '#F59E0B'}, # Amber
                            {'range': [70, 100], 'color': '#EF4444'} # Red
                        ],
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk summary card
                color_map = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
                border_color = color_map.get(risk, "#334155")
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 6px solid {border_color};">
                    <h4 style='margin-top:0px; color:{border_color};'>Risk Severity: {risk} Risk</h4>
                    <p style='margin-bottom:0px; font-size:15px;'>
                        <b>Model Prediction:</b> { "At-Risk of Churning" if warning else "Likely to Retain" }<br>
                        <b>Serving Node:</b> {source}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Retention Advice Card
                st.write("")
                st.subheader("📋 Actionable Retention Playbook")
                advice = []
                
                if risk == "High" or risk == "Medium":
                    if contract == "Month-to-month":
                        advice.append("⚠️ **Contract Vulnerability**: Customer is on a Month-to-month plan. Offer a discounted **One-Year or Two-Year Contract** contract to secure loyalty.")
                    if internet_service != "No":
                        if tech_support == "No":
                            advice.append("🛠️ **Support Friction**: Customer has no Tech Support. Pitch an premium **Tech Support add-on package** for troubleshooting comfort.")
                        if online_security == "No":
                            advice.append("🔒 **Security Gap**: Customer lacks Online Security. Suggest free trial of **Online Security Suite** to increase retention surface area.")
                    if tenure < 6:
                        advice.append("⏳ **Early Churn Risk**: Customer is in their critical onboarding phase (under 6 months). Schedule a **customer success check-in call**.")
                    if monthly_charges > 85.0:
                        advice.append("💵 **Price Sensitivity**: Customer monthly bill is high (${:.2f}). Propose value-tier bundles or a **10% customer loyalty discount**.".format(monthly_charges))
                    
                    if not advice:
                        advice.append("🤝 **General Retention Reach-out**: Send a loyalty satisfaction email and check NPS feedback.")
                else:
                    advice.append("✅ **Healthy Customer Profile**: Customer indicators are fully robust. Maintain standard check-in cycles and continue standard service delivery.")
                
                for item in advice:
                    st.write(item)
            else:
                st.warning("Prediction pipeline offline. Please complete training first.")
        else:
            st.info("👈 Set customer parameters in the left sidebar/column and click 'Analyze Attrition Risk' to view probability scores.")

with tab2:
    st.subheader("📈 Offline Validation Performance (DVC pipeline stages)")
    st.write("These metrics represent evaluation scores compiled during the DVC pipeline run, evaluated against the test dataset.")
    
    metrics = load_metrics()
    if metrics:
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        # Cards
        with col_m1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <span style="color: #94A3B8; font-size:14px; text-transform: uppercase;">Accuracy</span>
                <h2 style="margin: 5px 0 0 0; color: #6366F1;">{metrics.get('accuracy', 0.0):.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <span style="color: #94A3B8; font-size:14px; text-transform: uppercase;">F1 Score</span>
                <h2 style="margin: 5px 0 0 0; color: #10B981;">{metrics.get('f1_score', 0.0):.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <span style="color: #94A3B8; font-size:14px; text-transform: uppercase;">Precision</span>
                <h2 style="margin: 5px 0 0 0; color: #3B82F6;">{metrics.get('precision', 0.0):.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <span style="color: #94A3B8; font-size:14px; text-transform: uppercase;">Recall</span>
                <h2 style="margin: 5px 0 0 0; color: #F59E0B;">{metrics.get('recall', 0.0):.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col_m5:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <span style="color: #94A3B8; font-size:14px; text-transform: uppercase;">ROC AUC</span>
                <h2 style="margin: 5px 0 0 0; color: #EC4899;">{metrics.get('roc_auc', 0.0):.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        # Draw Plotly figure
        st.write("")
        st.write("")
        fig_metrics = px.bar(
            x=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            y=[metrics.get('accuracy'), metrics.get('precision'), metrics.get('recall'), metrics.get('f1_score'), metrics.get('roc_auc')],
            labels={'x': 'Performance Metric', 'y': 'Percentage Score'},
            title="DVC Pipeline Evaluation Metrics",
            color=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            color_discrete_sequence=px.colors.qualitative.G10
        )
        fig_metrics.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
    else:
        st.warning("Metrics JSON file 'data/metrics.json' is missing. Please execute the DVC pipeline.")

with tab3:
    st.subheader("🔬 Real-Time Serving Request Distribution & Evidently Drift Alerts")
    st.write("This tab displays automated drift diagnostics, comparing serving distribution parameters against training reference baselines.")

    log_path = "data/serving_log.csv"
    if os.path.exists(log_path):
        serving_df = pd.read_csv(log_path)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <span style="color: #94A3B8; font-size:14px; text-transform: uppercase;">Production Logs Captured</span>
                <h2 style="margin: 5px 0 0 0; color: #6366F1;">{len(serving_df)} Requests</h2>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            churn_ratio = serving_df['churn_prediction'].mean() if 'churn_prediction' in serving_df.columns else 0.0
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <span style="color: #94A3B8; font-size:14px; text-transform: uppercase;">Average Predicted Churn Rate</span>
                <h2 style="margin: 5px 0 0 0; color: #EF4444;">{churn_ratio:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.subheader("📋 Production Serving Logs (Last 10 records)")
        st.dataframe(serving_df.tail(10), use_container_width=True)
        
        # Load drift report link or frame
        drift_report_html = "data/drift_report.html"
        st.write("")
        st.subheader("📊 evidently AI - Interactive Data Drift Dashboard")
        
        # Pull drift statistics via API request if available
        api_monitor_url = "http://localhost:8000/monitor"
        drift_stats = None
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(api_monitor_url)
                if response.status_code == 200:
                    drift_stats = response.json()
        except Exception:
            pass
            
        if drift_stats and drift_stats.get("status") == "success":
            st.success("Evidently AI Drift analysis executed successfully!")
            col_d1, col_d2, col_d3 = st.columns(3)
            summary = drift_stats["drift_summary"]
            with col_d1:
                st.metric("Drift Status", "Drift Detected" if drift_stats["drift_detected"] else "No Significant Drift", 
                          delta="Warning" if drift_stats["drift_detected"] else "Normal", delta_color="inverse")
            with col_d2:
                st.metric("Drifted Features", f"{summary['drifted_features_count']} of {summary['total_features_count']}")
            with col_d3:
                st.metric("Drift Ratio", f"{summary['share_of_drifted_features']:.1%}")
        
        if os.path.exists(drift_report_html):
            st.info("Interactive HTML drift report is generated and can be rendered below.")
            with open(drift_report_html, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=800, scrolling=True)
        else:
            st.warning("Evidently data drift dashboard HTML report is missing. Run integration tests or hit the '/monitor' API endpoint to compile it.")
    else:
        st.warning("No production serving request logs found at 'data/serving_log.csv'. Make predictions under the 'Real-Time Inference' tab to generate serving logs.")
