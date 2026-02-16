"""Pipeline page — video upload → YOLO → physics → Gemini → report."""

import streamlit as st
import sys, os, time, tempfile, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from api import (
    aggregate_biomarkers, generate_gemini_report, build_complete_report,
    log_analysis_result, build_raw_data,
)
from physics_engine import process_frames

try:
    from yolo_inference import load_models, process_video, is_loaded
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    def load_models(): return False
    def process_video(path, target_fps=10.0, output_video_path=None): return [], None
    def is_loaded(): return False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from charts import (
    entropy_chart, fluency_chart, fractal_chart, phase_space_chart,
    kinetic_energy_chart, symmetry_chart, confidence_timeline, confidence_gauge,
)

# --- Page Header ---
st.title("Analysis Pipeline")

mode = st.session_state.get("app_mode", "clinical")
is_training = mode == "training"

if is_training:
    st.info("**Training Mode** — Expert feedback will refine the physics engine after analysis.")

# --- Video Upload ---
st.subheader("1. Upload Video")
uploaded = st.file_uploader(
    "Select a neonatal video for analysis",
    type=["mp4", "mov", "webm", "avi"],
    key="pipeline_upload",
)

if uploaded:
    tab_orig, tab_skel = st.tabs(["Original Video", "Skeleton Overlay"])
    with tab_orig:
        st.video(uploaded)
    with tab_skel:
        if "annotated_video" in st.session_state:
            st.video(st.session_state.annotated_video, format="video/mp4")
        else:
            st.caption("Run analysis to generate skeleton overlay.")

# --- Training Mode: Config & Expert Inputs ---
config = dict(st.session_state.get("motion_config", {
    "sensitivity": 0.85, "windowSize": 30, "entropyThreshold": 0.4,
    "jerkThreshold": 5.0, "rhythmicityWeight": 0.7, "stiffnessThreshold": 0.6,
}))

if is_training:
    with st.expander("Physics Engine Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            config["sensitivity"] = st.slider("Sensitivity", 0.1, 1.0, config["sensitivity"], 0.05)
            config["windowSize"] = st.slider("Window Size", 10, 60, config["windowSize"], 5)
            config["entropyThreshold"] = st.slider("Entropy Threshold (r)", 0.05, 0.5, config["entropyThreshold"], 0.05)
        with col2:
            config["jerkThreshold"] = st.slider("Jerk Threshold", 1.0, 10.0, config["jerkThreshold"], 0.5)
            config["rhythmicityWeight"] = st.slider("Rhythmicity Weight", 0.0, 2.0, config["rhythmicityWeight"], 0.1)
            config["stiffnessThreshold"] = st.slider("Stiffness Threshold", 0.1, 2.0, config["stiffnessThreshold"], 0.1)
        st.session_state.motion_config = config

    st.subheader("Expert Ground Truth")
    gt_col1, gt_col2 = st.columns(2)
    with gt_col1:
        expert_diagnosis = st.selectbox(
            "Expected Classification",
            ["", "Normal", "Sarnat Stage I", "Sarnat Stage II", "Sarnat Stage III", "Seizures"],
            index=0,
        )
        st.session_state.expert_diagnosis = expert_diagnosis or None
    with gt_col2:
        expert_annotation = st.text_area("Clinical Notes", value=st.session_state.get("expert_annotation", ""))
        st.session_state.expert_annotation = expert_annotation

st.divider()

# --- Run Analysis ---
st.subheader("2. Run Analysis")

if not YOLO_AVAILABLE or not is_loaded():
    col_load, col_status = st.columns([1, 2])
    with col_load:
        if st.button("Load YOLO26 Model", type="primary"):
            with st.spinner("Loading YOLO26x-Pose model..."):
                success = load_models()
            if success:
                st.success("Model loaded.")
                st.rerun()
            else:
                st.error("Failed to load YOLO26 model. Check CUDA/GPU setup.")
    with col_status:
        st.caption("YOLO26 Pose model must be loaded before analysis.")

can_run = uploaded is not None and YOLO_AVAILABLE and is_loaded()

if st.button("Run Clinical Analysis", disabled=not can_run, type="primary", use_container_width=True):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = tmp.name

    try:
        # Stage 1: Ingestion
        progress = st.progress(0, text="Stage 1/4 — Video Ingestion")
        time.sleep(0.3)

        # Stage 2: YOLO Pose Estimation
        progress.progress(25, text="Stage 2/4 — YOLO26 Pose Estimation")
        annotated_path = temp_path + "_skeleton.mp4"
        with st.spinner("Running YOLO26x-Pose..."):
            skeleton_frames, annotated_path = process_video(
                temp_path, target_fps=10.0, output_video_path=annotated_path,
            )

        if len(skeleton_frames) < 10:
            st.error(f"Only {len(skeleton_frames)} frames extracted. Need at least 10 for analysis.")
            st.stop()

        st.caption(f"{len(skeleton_frames)} skeleton frames extracted.")

        # Stage 3: Physics Engine
        progress.progress(50, text="Stage 3/4 — Movement Lab (Physics Engine)")
        with st.spinner("Computing biomarkers..."):
            metrics, posture, seizure = process_frames(skeleton_frames, config)

        if not metrics:
            st.error("Physics engine returned no metrics.")
            st.stop()

        # Stage 4: Gemini Classification
        progress.progress(75, text="Stage 4/4 — AI Classification (Gemini)")
        with st.spinner("Generating clinical assessment..."):
            biomarkers = aggregate_biomarkers(
                metrics, extra={"frames_processed": len(skeleton_frames)}
            )
            ground_truth = st.session_state.get("expert_diagnosis") if is_training else None
            gemini_report = generate_gemini_report(biomarkers, posture=posture, seizure=seizure)
            report = build_complete_report(
                biomarkers, gemini_report, metrics, posture=posture, seizure=seizure
            )

        # Log result
        entry_id = log_analysis_result(
            biomarkers=biomarkers,
            gemini_response=gemini_report,
            ground_truth=ground_truth,
            metadata={
                "source": "streamlit_upload",
                "filename": uploaded.name,
                "frames_processed": len(skeleton_frames),
            },
            first_frame_skeleton=skeleton_frames[0] if skeleton_frames else None,
            timeline_data=metrics,
        )

        progress.progress(100, text="Analysis Complete")

        # Store results in session
        report["id"] = entry_id
        report["date"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        report["videoName"] = uploaded.name
        st.session_state.current_report = report
        st.session_state.selected_report = report
        st.session_state.raw_frames = skeleton_frames
        st.session_state.chart_data = metrics

        # Store annotated video in session for skeleton overlay tab
        if annotated_path and os.path.exists(annotated_path):
            with open(annotated_path, "rb") as vf:
                st.session_state.annotated_video = vf.read()
            os.remove(annotated_path)

        st.success(f"Analysis complete — **{report['classification']}** ({report['confidence']}% confidence)")

        # --- Training mode: refine config ---
        if is_training and st.session_state.get("expert_diagnosis"):
            st.divider()
            st.subheader("Training: Config Refinement")
            with st.spinner("AI is refining physics engine parameters..."):
                try:
                    from api import Client, types
                    import json as _json
                    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
                    if api_key and Client:
                        client = Client(api_key=api_key)
                        refine_prompt = f"""You are optimizing a physics-based motor assessment algorithm.
Current diagnosis: {report['classification']}. Expert diagnosis: {st.session_state.expert_diagnosis}.
Current config: {_json.dumps(config)}. Biomarkers: {_json.dumps(biomarkers, default=str)}.
Return ONLY a JSON object with the new MotionConfig parameters."""
                        resp = client.models.generate_content(
                            model='gemini-3-pro-preview', contents=refine_prompt,
                            config=types.GenerateContentConfig(response_mime_type='application/json'),
                        )
                        new_config = _json.loads(resp.text)
                        st.write("**Before:**", config)
                        st.write("**After:**", new_config)
                        if st.button("Apply Refined Config"):
                            st.session_state.motion_config = new_config
                            st.rerun()
                except Exception as e:
                    st.warning(f"Config refinement failed: {e}")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if "annotated_path" in dir() and annotated_path and os.path.exists(annotated_path):
            os.remove(annotated_path)

# --- Show Charts (if data exists) ---
chart_data = st.session_state.get("chart_data", [])
if chart_data:
    st.divider()
    st.subheader("3. Movement Metrics")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(entropy_chart(chart_data), use_container_width=True)
    with c2:
        st.plotly_chart(fluency_chart(chart_data), use_container_width=True)
    with c3:
        st.plotly_chart(fractal_chart(chart_data), use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.plotly_chart(phase_space_chart(chart_data), use_container_width=True)
    with c5:
        st.plotly_chart(kinetic_energy_chart(chart_data), use_container_width=True)
    with c6:
        st.plotly_chart(symmetry_chart(chart_data), use_container_width=True)

    # Confidence timeline
    st.plotly_chart(confidence_timeline(chart_data), use_container_width=True)

    # Navigate to report
    if st.session_state.get("current_report"):
        if st.button("View Full Report", type="primary", use_container_width=True):
            st.session_state.selected_report = st.session_state.current_report
            st.switch_page("pages/2_Report.py")
