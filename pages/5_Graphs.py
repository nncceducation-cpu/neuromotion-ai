"""Graphs/Trends page — longitudinal metric trends across all analyses."""

import streamlit as st
import sys, os
import plotly.graph_objects as go
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from api import _read_jsonl, _jsonl_entry_to_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charts import COLORS, _base_layout

st.title("Trends")

# --- Load All Reports ---
entries = _read_jsonl()
entries.sort(key=lambda x: x.get("timestamp", ""))
reports = [_jsonl_entry_to_report(e) for e in entries]

st.caption(f"{len(reports)} analyses over time")

if len(reports) < 2:
    st.info("Run at least 2 analyses to see trends.")
    if st.button("Go to Analysis"):
        st.switch_page("pages/1_Pipeline.py")
    st.stop()

# Build chart data: one point per report
chart_data = []
for i, r in enumerate(reports):
    raw = r.get("rawData", {})
    date_str = r.get("date", "")
    try:
        dt = datetime.fromisoformat(date_str)
        date_label = dt.strftime("%m/%d/%Y")
    except Exception:
        date_label = date_str[:10]

    chart_data.append({
        "index": i + 1,
        "date": date_label,
        "label": r.get("videoName", "Unknown"),
        "classification": r.get("classification", "Normal"),
        "confidence": r.get("confidence", 0),
        "entropy": raw.get("entropy", 0),
        "fluency": raw.get("fluency", 0),
        "complexity": raw.get("complexity", 0),
        "kinetic_energy": raw.get("avg_kinetic_energy", 0),
    })

# --- Metric Definitions ---
METRICS = [
    {"key": "entropy", "label": "Entropy", "desc": "Movement Predictability", "color": "#171717"},
    {"key": "fluency", "label": "Fluency (Jerk)", "desc": "Movement Smoothness", "color": "#525252"},
    {"key": "complexity", "label": "Fractal Dimension", "desc": "Movement Complexity", "color": "#737373"},
    {"key": "kinetic_energy", "label": "Kinetic Energy", "desc": "Movement Vigor", "color": "#404040"},
    {"key": "confidence", "label": "Confidence", "desc": "AI Certainty", "color": "#171717"},
]

# --- Trend Charts ---
cols = st.columns(2)

for i, metric in enumerate(METRICS):
    dates = [d["date"] for d in chart_data]
    values = [d[metric["key"]] for d in chart_data]
    labels = [d["label"] for d in chart_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values, mode="lines+markers",
        name=metric["label"],
        line=dict(color=metric["color"], width=2),
        marker=dict(size=6, color=metric["color"]),
        text=labels,
        hovertemplate="%{text}<br>%{x}<br>%{y:.3f}<extra></extra>",
    ))

    y_range = [0, 100] if metric["key"] == "confidence" else None
    fig.update_layout(**_base_layout(
        title=dict(text=f"{metric['label']} — {metric['desc']}", font=dict(size=12)),
        height=280,
        yaxis=dict(range=y_range, showgrid=True, gridcolor=COLORS["grid"]),
    ))

    with cols[i % 2]:
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Classification History Table ---
st.subheader("Classification History")

table_rows = []
for d in chart_data:
    table_rows.append({
        "#": d["index"],
        "Date": d["date"],
        "Video": d["label"],
        "Classification": d["classification"],
        "Confidence": f"{d['confidence']}%",
        "Entropy": f"{d['entropy']:.3f}",
        "Jerk": f"{d['fluency']:.2f}",
    })

st.dataframe(table_rows, use_container_width=True, hide_index=True)
