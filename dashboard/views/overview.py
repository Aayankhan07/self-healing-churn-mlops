"""
Executive overview: KPIs, risk distribution, and trend charts.
"""

from datetime import datetime, timedelta

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import API_URL


def render(current_domain, current_domain_id, health, drift):
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
