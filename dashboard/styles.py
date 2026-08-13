"""
Dashboard styling.

The whole stylesheet lives here as one constant so page modules stay readable;
they call `apply()` once at startup rather than carrying 200 lines of CSS.
"""

import streamlit as st

STYLES = """
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
"""


def apply() -> None:
    """Inject the stylesheet. Call once, before rendering any page."""
    st.markdown(STYLES, unsafe_allow_html=True)
