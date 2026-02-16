"""Report page — full clinical assessment display with metrics, charts, and expert correction."""

import streamlit as st
import sys, os, io, csv, json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from api import _read_jsonl, _write_jsonl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charts import confidence_gauge, mini_chart

# --- Load Report ---
report = st.session_state.get("selected_report")

if not report:
    st.warning("No report selected. Go to Dashboard and select a report.")
    if st.button("Back to Dashboard"):
        st.switch_page("pages/0_Dashboard.py")
    st.stop()

# --- Header ---
col_back, col_title = st.columns([1, 5])
with col_back:
    if st.button("Back"):
        st.switch_page("pages/0_Dashboard.py")

st.title(f"{report.get('videoName', 'Analysis Report')}")

# --- Classification + Confidence ---
hdr1, hdr2, hdr3 = st.columns([2, 1, 1])

with hdr1:
    cls = report.get("classification", "Normal")
    conf = report.get("confidence", 0)

    # Expert override indicator
    correction = report.get("expertCorrection")
    if correction:
        st.success(f"**Expert Override:** {correction.get('correctClassification', cls)}")
        st.caption(f"Original AI: {cls} | Corrected by: {correction.get('clinicianName', 'Unknown')}")
    else:
        st.metric("Classification", cls)

with hdr2:
    st.plotly_chart(confidence_gauge(conf), use_container_width=True)

with hdr3:
    seizure_detected = report.get("seizureDetected", False)
    if seizure_detected:
        st.error(f"Seizure Detected: **{report.get('seizureType', 'Unknown')}**")
    else:
        st.success("No Seizure Detected")

# Differential Alert
diff_alert = report.get("differentialAlert")
if diff_alert:
    st.warning(f"**Differential Alert:** {diff_alert}")

st.divider()

# --- Raw Data Panels ---
raw = report.get("rawData", {})
posture = raw.get("posture", {})
seizure_data = raw.get("seizure", {})
timeline = report.get("timelineData", [])

tab_posture, tab_seizure, tab_physics, tab_charts = st.tabs(
    ["Posture & Tone", "Seizure Risk", "Physics Analysis", "Timeline Charts"]
)

with tab_posture:
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("Tone", posture.get("tone_label", "Normal"))
        st.metric("Shoulder Flexion", f"{posture.get('shoulder_flexion_index', 0):.2f}")
        st.metric("Hip Flexion", f"{posture.get('hip_flexion_index', 0):.2f}")
    with p2:
        st.metric("Symmetry Score", f"{posture.get('symmetry_score', 1.0):.2f}")
        st.metric("Frog Leg Score", f"{posture.get('frog_leg_score', 0):.2f}")
        st.metric("Spontaneous Activity", f"{posture.get('spontaneous_activity', 0):.3f}")
    with p3:
        st.metric("Sustained Posture", f"{posture.get('sustained_posture_score', 0):.2f}")
        st.metric("Arousal Index", f"{posture.get('arousal_index', 0):.3f}")
        st.metric("State Transition Prob", f"{posture.get('state_transition_probability', 0):.2f}")

    if posture.get("crying_index", 0) > 0 or posture.get("eye_openness_index", 0) > 0:
        st.caption(f"Crying Index: {posture.get('crying_index', 0):.2f} | Eye Openness: {posture.get('eye_openness_index', 0):.2f}")
    else:
        st.caption("Crying/Eye metrics unavailable (YOLO lacks mouth/eyelid keypoints)")

with tab_seizure:
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Rhythmicity Score", f"{seizure_data.get('rhythmicity_score', 0):.3f}")
        st.metric("Stiffness Score", f"{seizure_data.get('stiffness_score', 0):.3f}")
    with s2:
        st.metric("Dominant Frequency", f"{seizure_data.get('dominant_frequency', 0):.2f} Hz")
        st.metric("Limb Synchrony", f"{seizure_data.get('limb_synchrony', 0):.3f}")
    with s3:
        st.metric("Eye Deviation", f"{seizure_data.get('eye_deviation_score', 0):.3f}")
        calc_type = seizure_data.get("calculated_type", "None")
        if calc_type != "None":
            st.error(f"Calculated Type: **{calc_type}**")
        else:
            st.success("Calculated Type: None")

with tab_physics:
    ph1, ph2, ph3 = st.columns(3)
    with ph1:
        st.metric("Entropy", f"{raw.get('entropy', 0):.3f}")
        st.metric("Fluency (Jerk)", f"{raw.get('fluency', 0):.3f}")
        st.metric("Complexity (Fractal)", f"{raw.get('complexity', 0):.3f}")
    with ph2:
        st.metric("Kinetic Energy", f"{raw.get('avg_kinetic_energy', 0):.3f}")
        st.metric("Root Stress", f"{raw.get('avg_root_stress', 0):.3f}")
        st.metric("Bilateral Symmetry", f"{raw.get('avg_bilateral_symmetry', 0):.3f}")
    with ph3:
        st.metric("Head Stability", f"{raw.get('avg_head_stability', 0):.3f}")
        st.metric("Lower Limb KE", f"{raw.get('avg_lower_limb_ke', 0):.3f}")
        st.metric("Angular Jerk", f"{raw.get('avg_angular_jerk', 0):.3f}")

    # Additional computed metrics
    st.caption(f"Variability Index: {raw.get('variabilityIndex', 0):.4f} | CS Risk Score: {raw.get('csRiskScore', 0):.4f}")

with tab_charts:
    if timeline:
        from charts import (
            entropy_chart, fluency_chart, fractal_chart,
            kinetic_energy_chart, symmetry_chart, confidence_timeline,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(entropy_chart(timeline), use_container_width=True)
            st.plotly_chart(fractal_chart(timeline), use_container_width=True)
            st.plotly_chart(kinetic_energy_chart(timeline), use_container_width=True)
        with c2:
            st.plotly_chart(fluency_chart(timeline), use_container_width=True)
            st.plotly_chart(symmetry_chart(timeline), use_container_width=True)
            st.plotly_chart(confidence_timeline(timeline), use_container_width=True)
    else:
        st.info("No timeline data available for this report.")

st.divider()

# --- Clinical Analysis ---
st.subheader("Clinical Analysis")
analysis = report.get("clinicalAnalysis", "")
if analysis:
    st.markdown(analysis)

# --- Recommendations ---
recommendations = report.get("recommendations", [])
if recommendations:
    st.subheader("Recommendations")
    for rec in recommendations:
        st.markdown(f"- {rec}")

st.divider()

# --- Actions ---
act1, act2, act3 = st.columns(3)

with act1:
    # CSV Export
    if timeline:
        headers = ["timestamp", "entropy", "fluency_velocity", "fluency_jerk",
                    "fractal_dim", "kinetic_energy", "bilateral_symmetry",
                    "root_stress", "head_stability", "avg_visibility"]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in timeline:
            writer.writerow(row)
        st.download_button(
            "Export CSV", buf.getvalue(),
            f"neuromotion_{report.get('id', 'report')}.csv", "text/csv",
        )

with act2:
    # JSON Export
    st.download_button(
        "Export JSON", json.dumps(report, indent=2, default=str),
        f"neuromotion_{report.get('id', 'report')}.json", "application/json",
    )

with act3:
    if st.button("Back to Dashboard", key="back_bottom"):
        st.switch_page("pages/0_Dashboard.py")

# --- Expert Correction ---
st.divider()
st.subheader("Expert Correction")

if not report.get("expertCorrection"):
    with st.form("correction_form"):
        corr_cls = st.selectbox(
            "Correct Classification",
            ["Normal", "Sarnat Stage I", "Sarnat Stage II", "Sarnat Stage III", "Seizures"],
        )
        corr_notes = st.text_area("Clinical Reasoning")
        corr_name = st.text_input("Clinician Name", value=st.session_state.user.get("name", ""))

        if st.form_submit_button("Save Correction", type="primary"):
            correction = {
                "correctClassification": corr_cls,
                "notes": corr_notes,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "clinicianName": corr_name,
            }

            # Update JSONL entry
            entries = _read_jsonl()
            for entry in entries:
                if entry.get("id") == report.get("id"):
                    entry["expert_correction"] = correction
                    entry["ground_truth"] = corr_cls
                    entry["doctor_notes"] = corr_notes
                    entry["validated_at"] = datetime.now(timezone.utc).isoformat()
                    break
            _write_jsonl(entries)

            # Update Gemini cache
            try:
                from api import add_correction as api_add_correction
                api_add_correction(
                    biomarkers={},
                    ai_classification=report.get("classification", "Unknown"),
                    doctor_classification=corr_cls,
                    doctor_notes=corr_notes,
                )
            except Exception:
                pass

            report["expertCorrection"] = correction
            st.session_state.selected_report = report
            st.success("Correction saved.")
            st.rerun()
else:
    ec = report["expertCorrection"]
    st.info(f"**Already corrected:** {ec.get('correctClassification')} by {ec.get('clinicianName', 'Unknown')}")
    if ec.get("notes"):
        st.caption(f"Notes: {ec['notes']}")
