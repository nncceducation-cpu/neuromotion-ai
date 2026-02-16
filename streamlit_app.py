"""
NeuroMotion AI — Streamlit Frontend
Replaces the React/TypeScript UI with a pure-Python Streamlit application.
"""

import streamlit as st
import sys
import os

# Add server directory to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

st.set_page_config(
    page_title="NeuroMotion AI",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
def init_state():
    defaults = {
        "user": {"id": "default-clinician", "name": "Clinical Specialist", "email": "lab@neuromotion.ai"},
        "app_mode": "clinical",
        "current_report": None,
        "selected_report": None,
        "raw_frames": [],
        "chart_data": [],
        "motion_config": {
            "sensitivity": 0.85, "windowSize": 30, "entropyThreshold": 0.4,
            "jerkThreshold": 5.0, "rhythmicityWeight": 0.7, "stiffnessThreshold": 0.6,
        },
        "reports_to_compare": [],
        "expert_diagnosis": None,
        "expert_annotation": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# --- Navigation ---
dashboard = st.Page("pages/0_Dashboard.py", title="Dashboard", icon=":material/home:", default=True)
pipeline = st.Page("pages/1_Pipeline.py", title="Analysis Pipeline", icon=":material/play_circle:")
report = st.Page("pages/2_Report.py", title="Report", icon=":material/description:")
comparison = st.Page("pages/3_Comparison.py", title="Comparison", icon=":material/compare_arrows:")
validation = st.Page("pages/4_Validation.py", title="Validation", icon=":material/fact_check:")
graphs = st.Page("pages/5_Graphs.py", title="Trends", icon=":material/show_chart:")

pg = st.navigation([dashboard, pipeline, report, comparison, validation, graphs])

# --- Sidebar ---
with st.sidebar:
    st.markdown("### NeuroMotion AI")
    user = st.session_state.user
    st.caption(f"{user['name']}")

    st.divider()
    mode = st.radio("Mode", ["Clinical", "Training"], horizontal=True,
                    index=0 if st.session_state.app_mode == "clinical" else 1)
    st.session_state.app_mode = mode.lower()

pg.run()
