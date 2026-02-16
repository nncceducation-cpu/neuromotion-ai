"""Comparison page — multi-dataset comparison with charts, AI analysis, and chat."""

import streamlit as st
import sys, os, csv, io, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from api import _read_jsonl, _jsonl_entry_to_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charts import (
    comparison_entropy_chart, comparison_phase_space, comparison_bar_chart,
    CHART_PALETTE,
)

st.title("Comparison")

# --- Dataset Management ---

def compute_stats(data):
    """Compute per-metric statistics for a dataset."""
    metric_keys = [
        "entropy", "fluency_velocity", "fluency_jerk", "fractal_dim",
        "kinetic_energy", "root_stress", "bilateral_symmetry",
        "lower_limb_kinetic_energy", "angular_jerk", "head_stability", "com_velocity",
    ]
    stats = {}
    for key in metric_keys:
        values = [float(d.get(key, 0)) for d in data]
        n = len(values)
        if n == 0:
            stats[key] = {"mean": 0, "min": 0, "max": 0, "std": 0}
            continue
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        stats[key] = {
            "mean": round(mean, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std": round(math.sqrt(variance), 4),
        }
    return stats


if "comparison_datasets" not in st.session_state:
    st.session_state.comparison_datasets = []

# Load from reports_to_compare (from Dashboard selection)
if st.session_state.get("reports_to_compare"):
    for r in st.session_state.reports_to_compare:
        timeline = r.get("timelineData", [])
        if timeline and not any(
            d["label"] == r.get("videoName", "Unknown") for d in st.session_state.comparison_datasets
        ):
            st.session_state.comparison_datasets.append({
                "label": r.get("videoName", "Unknown"),
                "data": timeline,
                "stats": compute_stats(timeline),
            })
    st.session_state.reports_to_compare = []

# --- Add Data Sources ---
st.subheader("Datasets")

add_method = st.radio("Add data from:", ["CSV Upload", "Saved Reports"], horizontal=True)

if add_method == "CSV Upload":
    csv_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    if csv_files:
        for f in csv_files:
            if any(d["label"] == f.name for d in st.session_state.comparison_datasets):
                continue
            content = f.getvalue().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            data = []
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except (ValueError, TypeError):
                        parsed[k] = v
                data.append(parsed)
            if data:
                st.session_state.comparison_datasets.append({
                    "label": f.name,
                    "data": data,
                    "stats": compute_stats(data),
                })
        st.rerun()
else:
    entries = _read_jsonl()
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    reports = [_jsonl_entry_to_report(e) for e in entries]
    reports_with_timeline = [r for r in reports if r.get("timelineData")]
    if reports_with_timeline:
        selected_names = st.multiselect(
            "Select reports to compare",
            [r.get("videoName", r.get("id", "Unknown")) for r in reports_with_timeline],
        )
        if st.button("Add Selected"):
            for name in selected_names:
                match = next((r for r in reports_with_timeline if r.get("videoName", r.get("id")) == name), None)
                if match and not any(d["label"] == name for d in st.session_state.comparison_datasets):
                    tl = match["timelineData"]
                    st.session_state.comparison_datasets.append({
                        "label": name,
                        "data": tl,
                        "stats": compute_stats(tl),
                    })
            st.rerun()
    else:
        st.info("No reports with timeline data available. Run analyses first.")

datasets = st.session_state.comparison_datasets

# Show loaded datasets
if datasets:
    cols = st.columns(len(datasets))
    for i, ds in enumerate(datasets):
        with cols[i]:
            st.markdown(f"**{ds['label']}**")
            st.caption(f"{len(ds['data'])} frames")
            if st.button("Remove", key=f"rm_{i}"):
                st.session_state.comparison_datasets.pop(i)
                st.rerun()

    st.divider()

    # --- Charts ---
    st.subheader("Time Series")
    st.plotly_chart(comparison_entropy_chart(datasets), use_container_width=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(comparison_phase_space(datasets), use_container_width=True)
    with ch2:
        st.plotly_chart(
            comparison_bar_chart(datasets, "fluency_velocity", "Mean Velocity"),
            use_container_width=True,
        )

    # Additional bar charts
    bar1, bar2, bar3 = st.columns(3)
    with bar1:
        st.plotly_chart(comparison_bar_chart(datasets, "entropy", "Mean Entropy"), use_container_width=True)
    with bar2:
        st.plotly_chart(comparison_bar_chart(datasets, "kinetic_energy", "Mean KE"), use_container_width=True)
    with bar3:
        st.plotly_chart(comparison_bar_chart(datasets, "fluency_jerk", "Mean Jerk"), use_container_width=True)

    st.divider()

    # --- Statistics Table ---
    st.subheader("Detailed Statistics")
    metric_labels = {
        "entropy": "Entropy", "fluency_velocity": "Velocity", "fluency_jerk": "Jerk",
        "fractal_dim": "Fractal Dim", "kinetic_energy": "Kinetic Energy",
        "root_stress": "Root Stress", "bilateral_symmetry": "Symmetry",
    }
    table_data = []
    for key, label in metric_labels.items():
        row = {"Metric": label}
        for ds in datasets:
            s = ds.get("stats", {}).get(key, {})
            row[f"{ds['label']} (mean)"] = f"{s.get('mean', 0):.4f}"
            row[f"{ds['label']} (std)"] = f"{s.get('std', 0):.4f}"
        table_data.append(row)
    st.dataframe(table_data, use_container_width=True)

    st.divider()

    # --- AI Analysis ---
    st.subheader("AI Analysis")

    # Build dataset summary for AI
    dataset_summaries = ""
    for ds in datasets:
        dataset_summaries += f"\n--- {ds['label']} ---\n"
        for key, s in ds.get("stats", {}).items():
            dataset_summaries += f"  {key}: mean={s.get('mean', 0):.4f}, std={s.get('std', 0):.4f}, min={s.get('min', 0):.4f}, max={s.get('max', 0):.4f}\n"

    ai_col, chat_col = st.columns([2, 1])

    with ai_col:
        if st.button("Generate AI Report", type="primary"):
            try:
                from api import _get_gemini_client
                import google.genai.types as types
                client = _get_gemini_client()
                if client:
                    with st.spinner("Generating AI analysis..."):
                        prompt = f"""You are an expert Biomechanics Data Scientist.
Analyze the difference between the following motion sessions:

{dataset_summaries}

Provide a concise 3-paragraph summary:
1. Performance Comparison (Intensity, Kinetic Energy & Output)
2. Stability & Control Analysis (Root Stress & Entropy/Complexity)
3. Kinematic Variability & Smoothness.

Focus ONLY on physics, movement patterns, and data trends.
DO NOT provide medical diagnoses or clinical interpretations."""
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", contents=prompt,
                        )
                        st.session_state.comparison_ai_report = response.text or "No analysis generated."
                else:
                    st.error("Gemini AI not configured. Set GEMINI_API_KEY.")
            except Exception as e:
                st.error(f"AI analysis failed: {e}")

        if st.session_state.get("comparison_ai_report"):
            st.markdown(st.session_state.comparison_ai_report)

    with chat_col:
        st.markdown("**Ask about the data**")

        if "comparison_chat" not in st.session_state:
            st.session_state.comparison_chat = []

        # Show chat history
        for msg in st.session_state.comparison_chat:
            role = msg["role"]
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(msg["text"])

        question = st.chat_input("Ask a question about the data...")
        if question:
            st.session_state.comparison_chat.append({"role": "user", "text": question})
            try:
                from api import _get_gemini_client
                import google.genai.types as types
                client = _get_gemini_client()
                if client:
                    prompt = f"""Context: The user is comparing motion capture sessions.
Data Summary: {dataset_summaries}
User Question: "{question}"
Answer specifically using the data. Keep it brief (under 50 words)."""
                    response = client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a helpful AI Biomechanics Assistant."
                        ),
                    )
                    answer = response.text or "I couldn't generate a response."
                else:
                    answer = "Gemini AI not configured."
            except Exception as e:
                answer = f"Error: {e}"
            st.session_state.comparison_chat.append({"role": "ai", "text": answer})
            st.rerun()

    st.divider()

    # --- Automated Report ---
    st.subheader("Automated Report")
    if st.button("Generate Automated Report"):
        from api import generate_automated_comparison_report
        req_datasets = [{"label": ds["label"], "stats": ds["stats"]} for ds in datasets]
        st.session_state.comparison_auto_report = generate_automated_comparison_report(req_datasets)

    if st.session_state.get("comparison_auto_report"):
        st.text(st.session_state.comparison_auto_report)
        st.download_button(
            "Download Report",
            st.session_state.comparison_auto_report,
            "neuromotion_comparison.txt", "text/plain",
        )

else:
    st.info("Upload CSV files or select saved reports to begin comparison.")
