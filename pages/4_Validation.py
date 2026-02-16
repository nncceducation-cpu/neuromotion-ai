"""Validation page — review AI predictions and submit ground truth labels."""

import streamlit as st
import sys, os, json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from api import _read_jsonl, _write_jsonl

try:
    from api import add_correction
except ImportError:
    def add_correction(**kwargs): pass

st.title("Validation Review")
st.caption("Review AI predictions and provide ground truth labels for training.")

# --- Controls ---
ctrl1, ctrl2 = st.columns([3, 1])
with ctrl1:
    show_validated = st.toggle("Show validated cases", value=False)
with ctrl2:
    if st.button("Refresh"):
        st.rerun()

# --- Load Cases ---
entries = _read_jsonl()
if not show_validated:
    entries = [e for e in entries if not e.get("ground_truth")]
entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
entries = entries[:50]

st.caption(f"{len(entries)} {'total' if show_validated else 'pending'} cases")

if not entries:
    st.success("All caught up — no pending validations.")
    st.stop()

# --- Layout: Case List + Detail ---
list_col, detail_col = st.columns([1, 1])

with list_col:
    st.subheader("Cases")

    if "validation_selected_idx" not in st.session_state:
        st.session_state.validation_selected_idx = None

    for idx, case in enumerate(entries):
        pred = case.get("gemini_prediction", {})
        bio = case.get("biomarkers", {})
        meta = case.get("metadata", {})

        timestamp_str = case.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(timestamp_str)
            display_time = dt.strftime("%m/%d/%Y %H:%M")
        except Exception:
            display_time = timestamp_str[:19]

        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            with h1:
                st.caption(display_time)
                cls = pred.get("classification", "Unknown")
                conf = pred.get("confidence", 0)
                if isinstance(conf, float) and conf <= 1.0:
                    conf = round(conf * 100)
                st.markdown(f"`{cls}` — {conf}%")

                if meta.get("filename"):
                    st.caption(f"File: {meta['filename']}")

            with h2:
                if case.get("ground_truth"):
                    st.success("Validated", icon=":material/check:")

            # Biomarker summary
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Entropy", f"{bio.get('average_sample_entropy', 0):.2f}", label_visibility="collapsed")
                st.caption("Entropy")
            with b2:
                st.metric("Jerk", f"{bio.get('average_jerk', 0):.2f}", label_visibility="collapsed")
                st.caption("Jerk")
            with b3:
                frames = bio.get("frames_processed") or meta.get("frame_count", "N/A")
                st.metric("Frames", str(frames), label_visibility="collapsed")
                st.caption("Frames")

            if st.button("Select", key=f"sel_case_{idx}", use_container_width=True):
                st.session_state.validation_selected_idx = idx

with detail_col:
    sel_idx = st.session_state.get("validation_selected_idx")

    if sel_idx is not None and sel_idx < len(entries):
        case = entries[sel_idx]
        pred = case.get("gemini_prediction", {})
        bio = case.get("biomarkers", {})
        skeleton = case.get("first_frame_skeleton")

        st.subheader("Case Details")

        # Skeleton visualization
        if skeleton and skeleton.get("joints"):
            from charts import skeleton_chart
            st.plotly_chart(
                skeleton_chart(skeleton["joints"], title="YOLO26 Pose"),
                use_container_width=True,
            )
            if skeleton.get("note"):
                st.caption(skeleton["note"])
        else:
            st.info("No skeleton data available for this case.")

        # AI Prediction
        st.markdown("**AI Prediction**")
        with st.container(border=True):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"Classification: `{pred.get('classification', 'Unknown')}`")
            with d2:
                conf = pred.get("confidence", 0)
                if isinstance(conf, float) and conf <= 1.0:
                    conf = round(conf * 100)
                st.markdown(f"Confidence: **{conf}%**")

            reasoning = pred.get("reasoning") or pred.get("clinicalAnalysis", "")
            if reasoning:
                st.caption(reasoning)

            recs = pred.get("recommendations")
            if recs:
                if isinstance(recs, list):
                    recs = ", ".join(recs)
                st.caption(f"Recommendations: {recs}")

        # Biomarkers detail
        st.markdown("**Biomarkers**")
        bm1, bm2 = st.columns(2)
        with bm1:
            st.metric("Avg Entropy", f"{bio.get('average_sample_entropy', 0):.3f}")
            st.metric("Peak Entropy", f"{bio.get('peak_sample_entropy', 0):.3f}")
        with bm2:
            st.metric("Avg Jerk", f"{bio.get('average_jerk', 0):.3f}")
            frames = bio.get("frames_processed") or case.get("metadata", {}).get("frame_count", "N/A")
            st.metric("Frames", str(frames))

        st.divider()

        # Validation Form
        if not case.get("ground_truth"):
            st.markdown("**Doctor Validation**")
            with st.form(f"validate_{sel_idx}"):
                ground_truth = st.selectbox(
                    "Ground Truth Classification",
                    ["", "Normal", "Sarnat Stage I", "Sarnat Stage II",
                     "Sarnat Stage III", "Seizures", "Uncertain"],
                )
                doctor_notes = st.text_area("Notes (Optional)", placeholder="Clinical observations...")

                if st.form_submit_button("Submit Validation", type="primary"):
                    if not ground_truth:
                        st.error("Please select a classification.")
                    else:
                        # Update JSONL
                        all_entries = _read_jsonl()
                        for entry in all_entries:
                            if entry.get("timestamp") == case.get("timestamp"):
                                entry["ground_truth"] = ground_truth
                                entry["doctor_notes"] = doctor_notes or None
                                entry["validated_at"] = datetime.now(timezone.utc).isoformat()
                                break
                        _write_jsonl(all_entries)

                        # Update Gemini cache
                        ai_cls = pred.get("classification", "Unknown")
                        try:
                            add_correction(
                                biomarkers=bio,
                                ai_classification=ai_cls,
                                doctor_classification=ground_truth,
                                doctor_notes=doctor_notes,
                            )
                        except Exception:
                            pass

                        st.success("Validation saved.")
                        st.session_state.validation_selected_idx = None
                        st.rerun()
        else:
            st.markdown("**Already Validated**")
            with st.container(border=True):
                st.markdown(f"Ground Truth: **{case['ground_truth']}**")
                if case.get("doctor_notes"):
                    st.caption(f"Notes: {case['doctor_notes']}")
    else:
        st.info("Select a case from the list to review.")
