import sys
import os

# 1. SETUP PATH: Add the current directory to sys.path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import asyncio
from contextlib import asynccontextmanager
import json
import time
import math
import uuid
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load .env from project root (one level up from server/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# 2. IMPORT: Shared models
from models import (
    MotionConfig, AnalysisReport, SavedReport, ExpertCorrection, User,
    LoginRequest, RegisterRequest, SaveReportRequest, CorrectionRequest,
    TrajectoryRequest, CompareAIReportRequest, CompareChatRequest,
    CompareStatsRequest, AutomatedReportRequest,
    AnalysisRequest, RefineConfigRequest, ValidationRequest,
)

# 3. IMPORT: Storage service
import storage as storage_service

# 4. IMPORT: Trajectory generator
import trajectory_generator

# 5. IMPORT: Physics engine
try:
    from physics_engine import process_frames
except ImportError:
    try:
        from .physics_engine import process_frames
    except ImportError:
        print("CRITICAL WARNING: 'physics_engine.py' not found. Ensure it is in the server/ directory.")
        process_frames = lambda frames, config: []

# 6. AI SDK IMPORT
try:
    from google.genai import Client
    import google.genai.types as types
except ImportError:
    print("Warning: Google GenAI SDK not found. Install google-genai.")
    Client = None  # type: ignore[assignment,misc]
    types = None  # type: ignore[assignment]

# 6b. YOLO26 POSE IMPORT (optional)
try:
    from yolo_inference import load_models, process_video, is_loaded
    YOLO_AVAILABLE = True
except ImportError:
    print("Info: yolo_inference not available. YOLO26 Pose disabled, MediaPipe-only mode.")
    YOLO_AVAILABLE = False
    def load_models() -> bool:
        return False
    def process_video(video_path: str, target_fps: float = 10.0) -> tuple:
        return [], None
    def is_loaded() -> bool:
        return False

# 6c. GEMINI CONTEXT CACHE
try:
    from gemini_cache import get_cache_name, init_cache as init_gemini_cache, add_correction
except ImportError:
    print("Warning: gemini_cache not found. Context caching disabled.")
    def get_cache_name() -> str | None:
        return None
    def init_gemini_cache() -> str | None:
        return None
    def add_correction(biomarkers: dict, ai_classification: str, doctor_classification: str, doctor_notes: str | None = None) -> str | None:
        return None

# 7. INITIALIZE APP

@asynccontextmanager
async def lifespan(app):
    """Startup: load YOLO26 Pose model + Gemini context cache."""
    if YOLO_AVAILABLE:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, load_models)
        if success:
            print("YOLO26 Pose model loaded successfully at startup.")
        else:
            print("WARNING: YOLO26 Pose model failed to load. /upload_video will be unavailable.")
    else:
        print("YOLO26 Pose not installed. Running in MediaPipe-only mode.")

    loop = asyncio.get_event_loop()
    cache_name = await loop.run_in_executor(None, init_gemini_cache)
    if cache_name:
        print(f"Gemini context cache ready: {cache_name}")
    else:
        print("Gemini context cache not initialized (will fall back to full prompt).")

    yield

app = FastAPI(lifespan=lifespan)

# GPU semaphore: prevent concurrent inference
_gpu_semaphore = asyncio.Semaphore(1)

# 8. CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

JSONL_DIR = os.path.join(os.path.dirname(__file__), "analysis_logs")
JSONL_FILE = os.path.join(JSONL_DIR, "gemini_predictions.jsonl")


def _read_jsonl() -> List[Dict[str, Any]]:
    """Read all entries from the JSONL predictions file."""
    if not os.path.exists(JSONL_FILE):
        return []
    entries = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_jsonl(entries: List[Dict[str, Any]]):
    """Rewrite the entire JSONL predictions file."""
    os.makedirs(JSONL_DIR, exist_ok=True)
    with open(JSONL_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _jsonl_entry_to_report(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a JSONL entry into SavedReport shape for the frontend."""
    pred = entry.get("gemini_prediction") or {}
    bio = entry.get("biomarkers") or {}
    meta = entry.get("metadata") or {}

    expert_correction = None
    if entry.get("ground_truth"):
        expert_correction = {
            "correctClassification": entry["ground_truth"],
            "notes": entry.get("doctor_notes") or "",
            "timestamp": entry.get("validated_at") or entry.get("timestamp", ""),
            "clinicianName": "Validator",
        }
    if entry.get("expert_correction"):
        expert_correction = entry["expert_correction"]

    return {
        "id": entry.get("id") or entry.get("timestamp", str(uuid.uuid4())),
        "date": entry.get("timestamp", ""),
        "videoName": meta.get("filename") or "Unknown",
        "classification": pred.get("classification", "Normal"),
        "confidence": pred.get("confidence", 0),
        "seizureDetected": pred.get("seizureDetected", False),
        "seizureType": pred.get("seizureType", "None"),
        "differentialAlert": pred.get("differentialAlert"),
        "clinicalAnalysis": pred.get("clinicalAnalysis", ""),
        "recommendations": pred.get("recommendations", []),
        "rawData": build_raw_data(bio),
        "timelineData": entry.get("timelineData"),
        "expertCorrection": expert_correction,
    }


def log_analysis_result(biomarkers: Dict[str, Any], gemini_response: Dict[str, Any],
                        ground_truth: Optional[str] = None, metadata: Optional[Dict] = None,
                        first_frame_skeleton: Optional[Dict[str, Any]] = None,
                        timeline_data: Optional[List[Dict[str, Any]]] = None) -> str:
    """Logs analysis results to the JSONL file. Returns the generated entry id."""
    os.makedirs(JSONL_DIR, exist_ok=True)

    skeleton_with_labels = None
    if first_frame_skeleton and "joints" in first_frame_skeleton:
        coco_keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        skeleton_with_labels = {
            "timestamp": first_frame_skeleton.get("timestamp", 0),
            "joints": first_frame_skeleton["joints"],
            "keypoint_labels": coco_keypoint_names,
            "note": "All 17 COCO keypoints - verify YOLO26 detected pose correctly"
        }

    entry_id = str(uuid.uuid4())
    log_entry = {
        "id": entry_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "biomarkers": biomarkers,
        "gemini_prediction": gemini_response,
        "ground_truth": ground_truth,
        "doctor_notes": None,
        "first_frame_skeleton": skeleton_with_labels,
        "metadata": metadata or {},
        "timelineData": timeline_data,
    }

    try:
        with open(JSONL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"Logged analysis to {JSONL_FILE}")
    except Exception as e:
        print(f"Warning: Failed to log analysis: {e}")

    return entry_id


def build_raw_data(
    biomarkers: Dict[str, Any],
    posture: Optional[Dict[str, Any]] = None,
    seizure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the rawData object from aggregated biomarkers for frontend consumption."""
    return {
        "entropy": biomarkers.get("average_sample_entropy", 0),
        "fluency": biomarkers.get("average_jerk", 0),
        "complexity": biomarkers.get("average_fractal_dimension", 0),
        "variabilityIndex": biomarkers.get("variability_index", 0),
        "csRiskScore": biomarkers.get("cs_risk_score", 0),
        "avg_kinetic_energy": biomarkers.get("average_kinetic_energy", 0),
        "avg_root_stress": biomarkers.get("average_root_stress", 0),
        "avg_bilateral_symmetry": biomarkers.get("bilateral_symmetry_index", 0),
        "avg_lower_limb_ke": biomarkers.get("average_lower_limb_ke", 0),
        "avg_angular_jerk": biomarkers.get("angular_jerk_index", 0),
        "avg_head_stability": biomarkers.get("head_stability_index", 0),
        "avg_com_velocity": biomarkers.get("average_com_velocity", 0),
        "posture": posture or {
            "shoulder_flexion_index": 0, "hip_flexion_index": 0,
            "symmetry_score": 1.0, "tone_label": "Normal",
            "frog_leg_score": 0, "spontaneous_activity": 0,
            "sustained_posture_score": 0, "crying_index": 0,
            "eye_openness_index": 0, "arousal_index": 0,
            "state_transition_probability": 0,
        },
        "seizure": seizure or {
            "rhythmicity_score": 0, "stiffness_score": 0,
            "eye_deviation_score": 0, "dominant_frequency": 0,
            "limb_synchrony": 0, "calculated_type": "None",
        },
    }


def aggregate_biomarkers(metrics: List[Dict[str, Any]], source: str = "Python/YOLO26-Pipeline",
                         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Aggregate per-frame metrics into summary biomarkers for Gemini classification."""
    ents = [m.get('entropy', 0) for m in metrics]
    jerks = [m.get('fluency_jerk', 0) for m in metrics]
    fractals = [m.get('fractal_dim', 0) for m in metrics]
    kes = [m.get('kinetic_energy', 0) for m in metrics]
    rss = [m.get('root_stress', 0) for m in metrics]
    syms = [m.get('bilateral_symmetry', 0) for m in metrics]
    ll_kes = [m.get('lower_limb_kinetic_energy', 0) for m in metrics]
    ang_jerks = [m.get('angular_jerk', 0) for m in metrics]
    head_stabs = [m.get('head_stability', 0) for m in metrics]
    com_vels = [m.get('com_velocity', 0) for m in metrics]
    visibilities = [m.get('avg_visibility', 1.0) for m in metrics]

    # Confidence-weighted aggregation: high-confidence frames count more
    vis_weights = np.array(visibilities)
    vis_sum = vis_weights.sum()
    if vis_sum > 0:
        w = vis_weights / vis_sum
    else:
        w = np.ones(len(metrics)) / max(1, len(metrics))

    # Variability index: standard deviation of entropy
    variability_index = float(np.std(ents)) if ents else 0.0

    # CS Risk Score: autocorrelation of bilateral wrist signal at lag 1
    cs_risk_score = 0.0
    if len(ents) > 2:
        ents_arr = np.array(ents)
        ents_centered = ents_arr - np.mean(ents_arr)
        var = np.var(ents_arr)
        if var > 1e-10:
            cs_risk_score = float(np.correlate(ents_centered[:-1], ents_centered[1:])[0] / (var * (len(ents_arr) - 1)))

    biomarkers = {
        "average_sample_entropy": float(np.average(ents, weights=w)),
        "peak_sample_entropy": float(np.max(ents)),
        "average_jerk": float(np.average(jerks, weights=w)) if jerks else 0.0,
        "average_fractal_dimension": float(np.average(fractals, weights=w)),
        "peak_fractal_dimension": float(np.max(fractals)),
        "average_kinetic_energy": float(np.average(kes, weights=w)),
        "average_root_stress": float(np.average(rss, weights=w)),
        "bilateral_symmetry_index": float(np.average(syms, weights=w)),
        "average_lower_limb_ke": float(np.average(ll_kes, weights=w)),
        "angular_jerk_index": float(np.average(ang_jerks, weights=w)),
        "head_stability_index": float(np.average(head_stabs, weights=w)),
        "average_com_velocity": float(np.average(com_vels, weights=w)),
        "elbow_rom": float(metrics[-1].get('_elbow_rom', 0)) if metrics else 0.0,
        "knee_rom": float(metrics[-1].get('_knee_rom', 0)) if metrics else 0.0,
        "variability_index": variability_index,
        "cs_risk_score": cs_risk_score,
        "average_frame_confidence": float(np.mean(visibilities)),
        "low_confidence_frame_ratio": float(sum(1 for v in visibilities if v < 0.5) / max(1, len(visibilities))),
        "backend_source": source,
    }
    if extra:
        biomarkers.update(extra)
    return biomarkers


def build_complete_report(biomarkers: Dict[str, Any], gemini_report: Dict[str, Any],
                          metrics: List[Dict[str, Any]],
                          posture: Optional[Dict[str, Any]] = None,
                          seizure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a complete AnalysisReport-shaped dict from biomarkers + Gemini output + timeline."""
    raw_data = build_raw_data(biomarkers, posture=posture, seizure=seizure)
    return {
        "classification": gemini_report.get("classification", "Normal"),
        "confidence": gemini_report.get("confidence", 50),
        "seizureDetected": gemini_report.get("seizureDetected", False),
        "seizureType": gemini_report.get("seizureType", "None"),
        "differentialAlert": gemini_report.get("differentialAlert"),
        "rawData": raw_data,
        "clinicalAnalysis": gemini_report.get("clinicalAnalysis", gemini_report.get("reasoning", "")),
        "recommendations": gemini_report.get("recommendations", []),
        "timelineData": metrics,
    }


def generate_gemini_report(biomarkers: Dict[str, Any],
                           posture: Optional[Dict[str, Any]] = None,
                           seizure: Optional[Dict[str, Any]] = None):
    """Sends biomarkers + posture/seizure assessments to Gemini for clinical classification."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")

    if not api_key:
        return {"error": "API Key missing"}
    if Client is None or types is None:
        return {"error": "Google GenAI SDK not installed"}

    try:
        client = Client(api_key=api_key)

        analysis_instructions = """You are an expert Neonatal Neurologist and Computer Vision Specialist.
Your task is to analyze the provided BIOMARKER DATA (extracted from video) and generate a structured clinical assessment.

### DIAGNOSTIC CRITERIA (MODIFIED SARNAT & ILAE):

1. **NEONATAL ENCEPHALOPATHY (Modified Sarnat Score)**:
   - **Normal:** Alert, flexed posture, normal tone, smooth movements.
     - Markers: Moderate-High Entropy (0.3-0.6), Low Jerk (4-7), Moderate Fractal Dimension, Bilateral Symmetry >0.7, Good Head Stability (<2.0), Active Lower Limbs.
   - **Sarnat Stage I (Mild):** Hyperalert, stare, JITTERINESS, hyper-reflexia.
     - Markers: HIGH Jerk (>8.0), HIGH Kinetic Energy, High Angular Jerk, High Frequency movements.
   - **Sarnat Stage II (Moderate):** Lethargic/obtunded, HYPOTONIC (Frog-leg/Extended).
     - Markers: LOW Entropy (<0.3), LOW Kinetic Energy, Low Fractal Dimension (<1.2), Poor Bilateral Symmetry (<0.5), Poor Head Stability (>4.0), Low Lower Limb KE, Restricted ROM.
   - **Sarnat Stage III (Severe):** Comatose/stuporous, FLACCID tone.
     - Markers: NEAR-ZERO Entropy (<0.1), ZERO Kinetic Energy, Minimal movement, Near-zero CoM velocity.

2. **SEIZURE CLASSIFICATION RULES (STRICT - ILAE 2021)**:
   - **Clonic Seizure:** Rhythmic jerking in seizure band (1.5-5 Hz), high jerk with regularity.
     - Markers: Very HIGH Jerk (>10), HIGH Angular Jerk, LOW Bilateral Symmetry (lateralized), moderate-high entropy with periodic pattern.
   - **Tonic Seizure:** Sustained body stiffness, high root stress.
     - Markers: LOW Entropy with HIGH Root Stress, elevated kinetic energy plateau, Poor Head Stability, restricted ROM.
   - **Myoclonic Seizure:** Shock-like, isolated high-jerk events without sustained rhythm.
     - Markers: Extreme jerk spikes, low average but high peak entropy, brief asymmetry.

3. **DIFFERENTIAL DIAGNOSIS (MIMICS)**:
   - **Jitteriness vs Seizure:** Jitteriness = High Frequency (>5Hz), stimulus-sensitive, NO eye deviation, usually SYMMETRIC (high bilateral symmetry).
   - **Normal vs Sarnat I:** Both can have moderate entropy. Key differentiator: Sarnat I has jerk >8 and hyperkinetic energy.
   - **Asymmetry Alert:** Bilateral Symmetry <0.4 warrants investigation for focal pathology.

### BIOMARKER REFERENCE RANGES:
- **Sample Entropy:** 0.3-0.6 = normal variability, >0.7 = dysregulated/chaotic, <0.2 = overly rigid/suppressed
- **Jerk (fluency):** 4-7 = normal smooth movement, >8 = jerky/uncoordinated, <3 = hypokinetic/absent
- **Fractal Dimension:** 1.3-1.7 = normal complexity, <1.2 = low complexity (suppressed), >1.8 = chaotic
- **Kinetic Energy:** Total body KE (upper + lower limbs). Near-zero = minimal/absent movement.
- **Root Stress:** Hip midpoint velocity. High = sustained postural deviation or stiffness.
- **Bilateral Symmetry Index:** 0.7-1.0 = normal symmetric movement, <0.5 = concerning asymmetry (lateralized pathology)
- **Lower Limb KE:** Leg kinetic energy. Near-zero with active upper limbs = upper/lower dissociation.
- **Angular Jerk Index:** Smoothness of elbow joint rotations. High = jerky angular motion.
- **Head Stability Index:** Nose velocity relative to shoulders. <2.0 = good head control, >4.0 = poor head control/hypotonia.
- **CoM Velocity:** Whole-body center of mass velocity. Overall movement vigor.
- **Elbow/Knee ROM:** Joint range of motion in degrees. Low = restricted, high = full/excessive range.
- **Frame Confidence:** Average pose detection confidence across all frames. Low values (<0.7) indicate poor video quality, occlusion, or unreliable keypoints — reduce diagnostic confidence accordingly.
- **Low Confidence Ratio:** Fraction of frames with avg visibility < 0.5. High values (>0.3) mean many unreliable frames.
- **Variability Index:** Standard deviation of entropy over time. High values indicate irregular movement patterns.
- Normal infant baselines: entropy ~0.4+/-0.15, jerk ~5.5+/-1.5, symmetry ~0.8+/-0.1

### POSTURE ASSESSMENT FIELDS (when provided):
- **tone_label:** "Normal" / "Hypotonic" / "Hypertonic" — physics engine's heuristic classification
- **frog_leg_score:** 0-1 (higher = more hip abduction / splayed posture, typical of hypotonia)
- **shoulder_flexion_index / hip_flexion_index:** 0-1 normalized joint angles
- **sustained_posture_score:** 0-1 (higher = more static/lethargic posture)
- **arousal_index:** Magnitude of entropy spike above mean (higher = more state changes)
- **state_transition_probability:** Fraction of frames crossing mean KE (higher = more active state shifts)

### SEIZURE SIGNAL ANALYSIS FIELDS (when provided):
- **rhythmicity_score:** 0-1, fraction of spectral power in seizure band (1.5-5 Hz). >0.5 = strong rhythmic component
- **dominant_frequency:** Peak frequency from FFT. 1.5-5 Hz = seizure band, >5 Hz = jitteriness/tremor
- **stiffness_score:** 0-1, derived from entropy variance. High = rigid/sustained posture
- **limb_synchrony:** 0-1, cross-correlation of L/R wrist velocities. Low = lateralized, high = bilateral
- **calculated_type:** Rule-based seizure type from physics engine (Clonic/Tonic/Myoclonic/None) — use as supporting evidence

### FEW-SHOT EXAMPLES:

Example 1 - Normal:
Input: {"average_sample_entropy": 0.42, "average_jerk": 5.1, "average_fractal_dimension": 1.45, "average_kinetic_energy": 3.2, "average_root_stress": 0.8, "bilateral_symmetry_index": 0.82, "average_lower_limb_ke": 1.4, "angular_jerk_index": 450, "head_stability_index": 1.2, "average_com_velocity": 2.1, "elbow_rom": 65, "knee_rom": 55}
Output: {"classification": "Normal", "confidence": 88, "seizureDetected": false, "seizureType": "None", "differentialAlert": null, "clinicalAnalysis": "All biomarkers within normal ranges. Entropy of 0.42 indicates healthy movement variability. Bilateral symmetry 0.82 shows coordinated L/R movement. Lower limb KE 1.4 confirms active leg movement. Head stability 1.2 indicates good head control.", "recommendations": ["Continue routine monitoring"]}

Example 2 - Sarnat Stage II:
Input: {"average_sample_entropy": 0.18, "average_jerk": 2.1, "average_fractal_dimension": 1.1, "average_kinetic_energy": 0.5, "average_root_stress": 0.3, "bilateral_symmetry_index": 0.55, "average_lower_limb_ke": 0.1, "angular_jerk_index": 80, "head_stability_index": 4.5, "average_com_velocity": 0.4, "elbow_rom": 20, "knee_rom": 15}
Output: {"classification": "Sarnat Stage II", "confidence": 82, "seizureDetected": false, "seizureType": "None", "differentialAlert": null, "clinicalAnalysis": "Low entropy (0.18) with poor bilateral symmetry (0.55) indicate suppressed, asymmetric movement. Near-zero lower limb KE (0.1) and restricted ROM (elbow 20deg, knee 15deg) suggest hypotonia. Poor head stability (4.5) consistent with lethargy/obtundation in moderate HIE.", "recommendations": ["Recommend continuous EEG monitoring", "Consider MRI within 24-72 hours", "Assess for therapeutic hypothermia eligibility"]}

Example 3 - Seizures (Clonic):
Input: {"average_sample_entropy": 0.75, "average_jerk": 12.3, "average_fractal_dimension": 1.6, "average_kinetic_energy": 8.1, "average_root_stress": 2.5, "bilateral_symmetry_index": 0.35, "average_lower_limb_ke": 5.2, "angular_jerk_index": 2800, "head_stability_index": 5.1, "average_com_velocity": 6.3, "elbow_rom": 110, "knee_rom": 95}
Output: {"classification": "Seizures", "confidence": 76, "seizureDetected": true, "seizureType": "Clonic", "differentialAlert": null, "clinicalAnalysis": "Very high jerk (12.3) with elevated angular jerk (2800) and poor bilateral symmetry (0.35) indicate lateralized rhythmic jerking consistent with clonic seizure. High lower limb KE (5.2) shows leg involvement. Poor head stability (5.1) and excessive ROM suggest uncontrolled movement.", "recommendations": ["Urgent EEG correlation required", "Consider IV phenobarbital", "Continuous video-EEG monitoring"]}

**REQUIRED OUTPUT FORMAT (strict JSON):**
{
  "classification": "Normal" | "Sarnat Stage I" | "Sarnat Stage II" | "Sarnat Stage III" | "Seizures",
  "confidence": <integer 0-100>,
  "seizureDetected": <boolean>,
  "seizureType": "None" | "Clonic" | "Tonic" | "Myoclonic",
  "differentialAlert": "<string warning if mimic pattern detected, or null>",
  "clinicalAnalysis": "<detailed 2-4 sentence clinical explanation citing specific biomarker values>",
  "recommendations": ["<array>", "<of>", "<action items>"]
}

**IMPORTANT:**
- Confidence is 0-100 (integer), reflecting how clear-cut the pattern is
- Always cite specific biomarker values in clinicalAnalysis
- Use differentialAlert to flag mimics
- seizureDetected must be true only when classification is "Seizures"
- recommendations must be an array of strings
"""

        posture_block = ""
        if posture:
            posture_block = f"""
**POSTURE ASSESSMENT (physics engine):**
{json.dumps(posture, indent=2)}
"""

        seizure_block = ""
        if seizure:
            seizure_block = f"""
**SEIZURE SIGNAL ANALYSIS (FFT-derived from wrist kinematics):**
{json.dumps(seizure, indent=2)}
Note: 'calculated_type' is the physics engine's rule-based seizure classification.
Use it as supporting evidence alongside the biomarkers, not as ground truth.
"""

        user_query = f"""{analysis_instructions}

**NOW ANALYZE THIS CASE:**
Biomarkers: {json.dumps(biomarkers, indent=2)}
{posture_block}{seizure_block}"""

        cache_name = get_cache_name()
        if cache_name:
            config = types.GenerateContentConfig(
                cached_content=cache_name,
                response_mime_type='application/json',
            )
        else:
            config = types.GenerateContentConfig(
                response_mime_type='application/json',
            )

        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=user_query,
            config=config,
        )

        if response.text is None:
            return _gemini_error_response("Gemini returned empty response")
        result = json.loads(response.text)

        result.setdefault("classification", "Normal")
        result.setdefault("confidence", 50)
        result.setdefault("seizureDetected", result.get("classification") == "Seizures")
        result.setdefault("seizureType", "None")
        result.setdefault("differentialAlert", None)
        result.setdefault("clinicalAnalysis", result.pop("reasoning", "No analysis provided"))
        if isinstance(result.get("recommendations"), str):
            result["recommendations"] = [result["recommendations"]]
        result.setdefault("recommendations", [])
        if isinstance(result["confidence"], float) and result["confidence"] <= 1.0:
            result["confidence"] = round(result["confidence"] * 100)

        return result

    except Exception as e:
        print(f"Gemini Error: {e}")
        return _gemini_error_response(f"AI service error: {str(e)}")


def _gemini_error_response(reason: str) -> Dict[str, Any]:
    return {
        "classification": "Normal",
        "confidence": 0,
        "seizureDetected": False,
        "seizureType": "None",
        "differentialAlert": None,
        "clinicalAnalysis": reason,
        "recommendations": ["Manual review required"]
    }


def _get_gemini_client():
    """Get a Gemini client instance for comparison AI endpoints."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
    if not api_key or Client is None:
        return None
    return Client(api_key=api_key)


# ============================================================================
# EXISTING API ROUTES
# ============================================================================

@app.post("/analyze_frames")
async def analyze_frames_endpoint(request: AnalysisRequest):
    """Endpoint for frontend to send keypoints for Physics Processing + Gemini."""
    print(f"Received {len(request.frames)} frames for analysis")

    metrics, posture, seizure = process_frames(request.frames, request.config.model_dump())

    if not metrics:
        return {"metrics": [], "report": {"classification": "Normal", "confidence": 0,
                "clinicalAnalysis": "Insufficient motion data extracted", "recommendations": []}}

    biomarkers = aggregate_biomarkers(metrics)
    gemini_report = generate_gemini_report(biomarkers, posture=posture, seizure=seizure)
    report = build_complete_report(biomarkers, gemini_report, metrics, posture=posture, seizure=seizure)

    entry_id = log_analysis_result(
        biomarkers=biomarkers,
        gemini_response=gemini_report,
        ground_truth=None,
        metadata={"source": "frontend_mediapipe", "frame_count": len(request.frames)},
        first_frame_skeleton=request.frames[0] if request.frames else None,
        timeline_data=metrics,
    )

    return {"report": report, "metrics": metrics, "entry_id": entry_id}


@app.post("/upload_video")
async def upload_video_for_pose(file: UploadFile = File(...)):
    """Upload a video for server-side YOLO26 Pose processing."""
    if not YOLO_AVAILABLE or not is_loaded():
        raise HTTPException(status_code=503, detail="YOLO26 Pose model not available")

    config = MotionConfig(
        sensitivity=0.85, windowSize=30, entropyThreshold=0.4,
        jerkThreshold=5.0, rhythmicityWeight=0.7, stiffnessThreshold=0.6
    )

    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_filename, "wb") as buffer:
            buffer.write(contents)
        print(f"Processing video: {temp_filename} ({len(contents)} bytes)")

        async with _gpu_semaphore:
            loop = asyncio.get_event_loop()
            skeleton_frames, _ = await loop.run_in_executor(
                None, lambda: process_video(temp_filename, target_fps=10.0)
            )

        if len(skeleton_frames) < 10:
            return {"metrics": [], "report": {"classification": "Normal", "confidence": 0,
                    "clinicalAnalysis": "Insufficient pose data from video", "recommendations": []},
                    "frames_processed": len(skeleton_frames)}

        metrics, posture, seizure = process_frames(skeleton_frames, config.model_dump())
        if not metrics:
            return {"metrics": [], "report": {"classification": "Normal", "confidence": 0,
                    "clinicalAnalysis": "Physics engine returned no metrics", "recommendations": []},
                    "frames_processed": len(skeleton_frames)}

        biomarkers = aggregate_biomarkers(metrics, extra={"frames_processed": len(skeleton_frames)})
        gemini_report = generate_gemini_report(biomarkers, posture=posture, seizure=seizure)
        report = build_complete_report(biomarkers, gemini_report, metrics, posture=posture, seizure=seizure)

        entry_id = log_analysis_result(
            biomarkers=biomarkers,
            gemini_response=gemini_report,
            ground_truth=None,
            metadata={"source": "video_upload", "filename": file.filename,
                      "frames_processed": len(skeleton_frames)},
            first_frame_skeleton=skeleton_frames[0] if skeleton_frames else None,
            timeline_data=metrics,
        )

        return {"report": report, "metrics": metrics,
                "frames_processed": len(skeleton_frames), "entry_id": entry_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Upload video error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.get("/health")
def health_check():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
        device = f"cuda:0 ({gpu_name})" if gpu_available else "cpu"
    except ImportError:
        gpu_available = False
        device = "cpu"
    return {
        "status": "active",
        "model": "YOLO26x-Pose" if YOLO_AVAILABLE else "MediaPipe-only",
        "device": device,
        "yolo_loaded": YOLO_AVAILABLE and is_loaded() if YOLO_AVAILABLE else False,
    }


@app.get("/")
def root():
    """API status endpoint."""
    return {"message": "Neuromotion AI Backend is Running. Use 'python app.py' for the Streamlit UI."}


@app.post("/refine_config")
async def refine_config(request: RefineConfigRequest):
    """AI-powered physics engine parameter tuning."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    if Client is None or types is None:
        raise HTTPException(status_code=500, detail="Google GenAI SDK not installed")

    prompt = f"""You are a Senior Computer Vision Engineer optimizing a physics-based motor assessment algorithm for neonates.

### GOAL
The current algorithm misclassified a video. You must adjust the signal processing parameters (MotionConfig) so that the physics engine extracts biomarkers that lead to the CORRECT diagnosis ("{request.expert_diagnosis}").

### CONTEXT
- **Current Diagnosis**: {request.current_report.get('classification', 'Unknown') if request.current_report else 'Unknown'}
- **Expert Diagnosis**: "{request.expert_diagnosis}"
- **Expert Annotation**: "{request.annotation}"

### PHYSICS ENGINE LOGIC & PARAMETERS
1. **sensitivity** (0.1-1.0): Controls peak detection threshold.
2. **windowSize** (10-60 frames): Smoothing window for Entropy/Fractals.
3. **entropyThreshold** (0.05-0.5): The 'r' radius for Sample Entropy matching.
4. **rhythmicityWeight** (0.0-2.0): Multiplier for Seizure Rhythmicity Score.
5. **jerkThreshold** (1.0-10.0): Baseline offset for Smoothness/Fluency.
6. **stiffnessThreshold** (0.1-2.0): Variance divider for stiffness.

### CURRENT CONFIGURATION
{json.dumps(request.current_config, indent=2)}

### BIOMARKER SNAPSHOT
{json.dumps(request.current_report.get('rawData', {{}}) if request.current_report else 'N/A', indent=2)}

### OPTIMIZATION STRATEGY
- **Missed Seizure?** Boost 'sensitivity' & 'rhythmicityWeight'.
- **False Seizure (Jitteriness)?** Reduce 'rhythmicityWeight', Reduce 'windowSize', Increase 'jerkThreshold'.
- **Missed Lethargy (Sarnat II)?** Increase 'windowSize', Increase 'entropyThreshold'.
- **False Lethargy?** Decrease 'entropyThreshold', Decrease 'windowSize'.

OUTPUT JSON ONLY (The new MotionConfig object).
"""

    try:
        client = Client(api_key=api_key)
        config = types.GenerateContentConfig(response_mime_type='application/json')
        response = client.models.generate_content(
            model='gemini-3-pro-preview', contents=prompt, config=config,
        )
        if not response.text:
            raise HTTPException(status_code=500, detail="Empty response from AI")
        new_config = json.loads(response.text)
        return new_config
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")
    except Exception as e:
        print(f"Refine config error: {e}")
        fallback = dict(request.current_config)
        fallback["sensitivity"] = min(1.0, fallback.get("sensitivity", 0.85) * 1.05)
        fallback["rhythmicityWeight"] = fallback.get("rhythmicityWeight", 0.7) * 0.95
        return fallback


@app.post("/validate")
async def validate_prediction(request: ValidationRequest):
    """Endpoint for doctors to submit ground truth labels for AI predictions."""
    entries = _read_jsonl()
    if not entries:
        raise HTTPException(status_code=404, detail="No analysis logs found")

    matched_entry = None
    for entry in entries:
        if entry.get("timestamp") == request.timestamp:
            entry["ground_truth"] = request.ground_truth_classification
            entry["doctor_notes"] = request.doctor_notes
            entry["validated_at"] = datetime.now(timezone.utc).isoformat()
            matched_entry = entry
            break

    if matched_entry is None:
        raise HTTPException(status_code=404, detail=f"No analysis found with timestamp {request.timestamp}")

    _write_jsonl(entries)

    ai_classification = (matched_entry.get("gemini_prediction") or {}).get("classification", "Unknown")
    add_correction(
        biomarkers=matched_entry.get("biomarkers", {}),
        ai_classification=ai_classification,
        doctor_classification=request.ground_truth_classification,
        doctor_notes=request.doctor_notes,
    )

    return {
        "status": "success",
        "message": f"Ground truth label '{request.ground_truth_classification}' saved. Gemini cache updated."
    }


@app.get("/pending_validations")
async def get_pending_validations(limit: int = 50, include_validated: bool = False):
    """Fetch analysis cases that need doctor validation."""
    entries = _read_jsonl()

    if not include_validated:
        entries = [e for e in entries if not e.get("ground_truth")]

    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    entries = entries[:limit]

    return {
        "cases": entries,
        "total": len(entries),
        "showing": "all" if include_validated else "unvalidated_only"
    }


# ============================================================================
# NEW: AUTH ENDPOINTS (converted from services/storage.ts)
# ============================================================================

@app.post("/auth/login")
async def auth_login(request: LoginRequest):
    """Login with email + password. Returns user object + session token."""
    try:
        user, token = storage_service.login(request.email, request.password)
        return {"user": user.model_dump(), "token": token}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/auth/register")
async def auth_register(request: RegisterRequest):
    """Register a new user."""
    try:
        user = storage_service.register(request.name, request.email, request.password)
        # Auto-login after registration
        _, token = storage_service.login(request.email, request.password)
        return {"user": user.model_dump(), "token": token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    """Get current user from session token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization header")
    token = authorization.replace("Bearer ", "")
    user = storage_service.get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user.model_dump()


@app.post("/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(None)):
    """Invalidate session token."""
    if authorization:
        token = authorization.replace("Bearer ", "")
        storage_service.logout(token)
    return {"status": "success"}


# ============================================================================
# REPORT CRUD ENDPOINTS — backed by gemini_predictions.jsonl
# ============================================================================

@app.get("/reports/{user_id}")
async def get_reports(user_id: str):
    """Get all analysis reports from the JSONL log."""
    entries = _read_jsonl()
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return [_jsonl_entry_to_report(e) for e in entries]


@app.post("/reports/{user_id}")
async def save_report(user_id: str, request: SaveReportRequest):
    """No-op: analyses are already saved to JSONL during processing.
    Returns the most recent JSONL entry as a SavedReport for backward compat."""
    entries = _read_jsonl()
    if entries:
        return _jsonl_entry_to_report(entries[-1])
    return {}


@app.delete("/reports/{user_id}/{report_id}")
async def delete_report(user_id: str, report_id: str):
    """Delete a report entry from the JSONL log."""
    entries = _read_jsonl()
    original_len = len(entries)
    entries = [e for e in entries if e.get("id") != report_id and e.get("timestamp") != report_id]
    if len(entries) == original_len:
        raise HTTPException(status_code=404, detail="Report not found")
    _write_jsonl(entries)
    return {"status": "success"}


@app.post("/reports/{user_id}/{report_id}/correction")
async def save_correction(user_id: str, report_id: str, request: CorrectionRequest):
    """Save an expert correction to a JSONL entry."""
    entries = _read_jsonl()
    matched = None
    for entry in entries:
        if entry.get("id") == report_id or entry.get("timestamp") == report_id:
            entry["expert_correction"] = request.correction.model_dump()
            entry["ground_truth"] = request.correction.correctClassification
            entry["doctor_notes"] = request.correction.notes
            entry["validated_at"] = datetime.now(timezone.utc).isoformat()
            matched = entry
            break
    if not matched:
        raise HTTPException(status_code=404, detail="Report not found")
    _write_jsonl(entries)

    # Update Gemini cache
    ai_classification = (matched.get("gemini_prediction") or {}).get("classification", "Unknown")
    add_correction(
        biomarkers=matched.get("biomarkers", {}),
        ai_classification=ai_classification,
        doctor_classification=request.correction.correctClassification,
        doctor_notes=request.correction.notes,
    )

    return _jsonl_entry_to_report(matched)


@app.get("/training_examples")
async def get_training_examples():
    """Get all expert-corrected entries from JSONL (last 10) for training."""
    entries = _read_jsonl()
    examples = []
    for e in entries:
        correction = e.get("expert_correction")
        if correction:
            report = _jsonl_entry_to_report(e)
            examples.append({
                "inputs": report.get("rawData"),
                "groundTruth": correction,
            })
    return examples[:10]


@app.get("/learned_stats/{user_id}")
async def get_learned_stats(user_id: str):
    """Get aggregated correction stats from JSONL."""
    entries = _read_jsonl()
    corrections = [e for e in entries if e.get("expert_correction") or e.get("ground_truth")]
    by_category: Dict[str, int] = {}
    for e in corrections:
        cat = (e.get("expert_correction") or {}).get("correctClassification") or e.get("ground_truth", "Unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    return {"totalLearned": len(corrections), "breakdown": by_category}


# ============================================================================
# NEW: TRAJECTORY ENDPOINTS (converted from constants.ts)
# ============================================================================

@app.post("/generate_trajectory")
async def generate_trajectory_endpoint(request: TrajectoryRequest):
    """Generate skeleton frames for a clinical profile."""
    try:
        frames = trajectory_generator.generate_trajectory(
            profile_id=request.profile_id,
            frames=request.frames,
            seed=request.seed,
        )
        return {"frames": frames, "profile_id": request.profile_id, "count": len(frames)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/clinical_profiles")
async def get_clinical_profiles():
    """Get list of clinical profile metadata (without generator functions)."""
    return [
        {
            "id": p["id"],
            "label": p["label"],
            "features": p["features"],
        }
        for p in trajectory_generator.CLINICAL_PROFILES
    ]


@app.get("/profile_by_seed/{seed}")
async def get_profile_by_seed(seed: int):
    """Get a clinical profile deterministically from a seed value."""
    profile = trajectory_generator.get_profile_by_seed(seed)
    return {
        "id": profile["id"],
        "label": profile["label"],
        "features": profile["features"],
    }


# ============================================================================
# NEW: COMPARISON AI ENDPOINTS (converted from ComparisonView.tsx Gemini calls)
# ============================================================================

@app.post("/compare/ai_report")
async def compare_ai_report(request: CompareAIReportRequest):
    """Generate an AI biomechanics analysis report from dataset summaries."""
    client = _get_gemini_client()
    if not client or types is None:
        raise HTTPException(status_code=500, detail="Gemini AI not configured")

    prompt = f"""
      You are an expert Biomechanics Data Scientist.
      Analyze the difference between the following motion sessions based on these computed metrics:

      {request.dataset_summaries}

      Provide a concise 3-paragraph summary:
      1. Performance Comparison (Intensity, Kinetic Energy & Output)
      2. Stability & Control Analysis (Root Stress & Entropy/Complexity)
      3. Kinematic Variability & Smoothness.

      STRICT REQUIREMENT: Focus ONLY on the physics, movement patterns, and data trends.
      DO NOT provide any medical diagnoses, clinical interpretations (e.g. Sarnat stages, CP), or medical advice.

      Use professional technical language.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return {"report": response.text or "No analysis generated."}
    except Exception as e:
        print(f"Compare AI Report Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI analysis")


@app.post("/compare/chat")
async def compare_chat(request: CompareChatRequest):
    """Chat about comparison data using Gemini AI."""
    client = _get_gemini_client()
    if not client or types is None:
        raise HTTPException(status_code=500, detail="Gemini AI not configured")

    prompt = f"""
      Context: The user is looking at a dashboard comparing motion capture sessions.
      Data Summary:
      {request.dataset_summaries}

      User Question: "{request.question}"

      Answer the user specifically using the data provided. Keep it helpful, encouraging, and brief (under 50 words if possible).
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful AI Sports Science and Biomechanics Assistant."
            ),
        )
        return {"response": response.text or "I couldn't generate a response."}
    except Exception as e:
        print(f"Compare Chat Error: {e}")
        return {"response": "Error connecting to AI service."}


@app.post("/compare/stats")
async def compare_stats(request: CompareStatsRequest):
    """Calculate statistics for a dataset of movement metrics."""
    metric_keys = ['entropy', 'fluency_velocity', 'fluency_jerk', 'fractal_dim', 'kinetic_energy', 'root_stress']
    stats: Dict[str, Any] = {}

    for key in metric_keys:
        values = [float(d.get(key, 0)) for d in request.data]
        n = len(values)
        if n == 0:
            stats[key] = {"mean": 0, "min": 0, "max": 0, "std": 0}
            continue
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance)
        stats[key] = {
            "mean": mean,
            "min": min(values),
            "max": max(values),
            "std": std,
        }

    return stats


def generate_automated_comparison_report(datasets: List[Dict[str, Any]]) -> str:
    """Generate an automated comparison report from dataset statistics (sync)."""
    if not datasets:
        return {"report": ""}

    lines: List[str] = []
    date = datetime.now().strftime("%m/%d/%Y")

    lines.append("NEUROMOTION AI - COMPARATIVE CLINICAL REPORT")
    lines.append(f"Date: {date}")
    lines.append(f"Subjects: {', '.join(d.get('label', 'Unknown') for d in datasets)}")
    lines.append("")
    lines.append("----------------------------------------------------------------")
    lines.append("1. EXECUTIVE SUMMARY & DETAILED ANALYSIS")
    lines.append("----------------------------------------------------------------")

    # Find leaders
    max_entropy = datasets[0]
    min_entropy = datasets[0]
    max_jerk = datasets[0]

    for d in datasets:
        s = d.get("stats", {})
        if s.get("entropy", {}).get("mean", 0) > max_entropy.get("stats", {}).get("entropy", {}).get("mean", 0):
            max_entropy = d
        if s.get("entropy", {}).get("mean", 0) < min_entropy.get("stats", {}).get("entropy", {}).get("mean", 0):
            min_entropy = d
        if s.get("fluency_jerk", {}).get("mean", 0) > max_jerk.get("stats", {}).get("fluency_jerk", {}).get("mean", 0):
            max_jerk = d

    me_val = max_entropy.get("stats", {}).get("entropy", {}).get("mean", 0)
    lines.append(f"* COMPLEXITY LEADER: {max_entropy.get('label', 'Unknown')}")
    lines.append(f"  - Highest variability (Mean Entropy: {me_val:.3f}).")
    lines.append("  - Clinical significance: Indicates a richer motor repertoire and healthy corticospinal integrity.")
    lines.append("")

    mie_val = min_entropy.get("stats", {}).get("entropy", {}).get("mean", 0)
    if min_entropy.get("label") != max_entropy.get("label"):
        lines.append(f"* CONCERN FOR POVERTY OF MOVEMENT: {min_entropy.get('label', 'Unknown')}")
        lines.append(f"  - Lowest variability (Mean Entropy: {mie_val:.3f}).")
        lines.append("  - Clinical significance: May indicate lethargy, hypotonia, or encephalopathy (Sarnat II/III).")
        lines.append("")

    mj_val = max_jerk.get("stats", {}).get("fluency_jerk", {}).get("mean", 0)
    lines.append(f"* HIGHEST ACTIVITY INTENSITY: {max_jerk.get('label', 'Unknown')}")
    lines.append(f"  - Peak Jerk Index: {mj_val:.2f}")
    lines.append("  - Clinical significance: If excessive (>8.0), consider jitteriness, hyperexcitability, or tremors.")
    lines.append("")

    lines.append("----------------------------------------------------------------")
    lines.append("2. DETAILED BIOMARKER PROFILES")
    lines.append("----------------------------------------------------------------")

    for d in datasets:
        s = d.get("stats", {})
        e = s.get("entropy", {}).get("mean", 0)
        v = s.get("fluency_velocity", {}).get("mean", 0)
        j = s.get("fluency_jerk", {}).get("mean", 0)
        f = s.get("fractal_dim", {}).get("mean", 0)
        ke = s.get("kinetic_energy", {}).get("mean", 0)

        lines.append(f"SUBJECT: {d.get('label', 'Unknown')}")
        lines.append(f"  * Entropy (Complexity):   {e:.3f} [Norm: >0.6]")
        lines.append(f"  * Velocity (Activity):    {v:.3f}")
        lines.append(f"  * Jerk (Smoothness):      {j:.3f} [Norm: <7.0]")
        lines.append(f"  * Fractal Dim (Texture):  {f:.3f}")
        lines.append(f"  * Kinetic Energy (PhysX): {ke:.2f} J")

        impression = []
        if e < 0.4:
            impression.append("Markedly reduced complexity (Warning)")
        elif e < 0.6:
            impression.append("Mildly reduced complexity")
        else:
            impression.append("Normal complexity")

        if j > 8.0:
            impression.append("High frequency tremors detected")

        lines.append(f"  => INTERPRETATION: {', '.join(impression)}")
        lines.append("")

    if len(datasets) == 2:
        lines.append("----------------------------------------------------------------")
        lines.append(f"3. DIRECT COMPARISON ({datasets[0].get('label')} vs {datasets[1].get('label')})")
        lines.append("----------------------------------------------------------------")
        d1s = datasets[0].get("stats", {})
        d2s = datasets[1].get("stats", {})

        e_diff = d1s.get("entropy", {}).get("mean", 0) - d2s.get("entropy", {}).get("mean", 0)
        v_diff = d1s.get("fluency_velocity", {}).get("mean", 0) - d2s.get("fluency_velocity", {}).get("mean", 0)
        j_diff = d1s.get("fluency_jerk", {}).get("mean", 0) - d2s.get("fluency_jerk", {}).get("mean", 0)

        d2e = d2s.get("entropy", {}).get("mean", 1) or 1
        d2v = d2s.get("fluency_velocity", {}).get("mean", 1) or 1
        d2j = d2s.get("fluency_jerk", {}).get("mean", 1) or 1

        lines.append(f"* ENTROPY: {datasets[0].get('label')} is {abs(e_diff / d2e * 100):.1f}% {'more' if e_diff > 0 else 'less'} complex.")
        lines.append(f"* VELOCITY: {datasets[0].get('label')} is {abs(v_diff / d2v * 100):.1f}% {'faster' if v_diff > 0 else 'slower'}.")
        lines.append(f"* JERK: {datasets[0].get('label')} is {abs(j_diff / d2j * 100):.1f}% {'jitterier' if j_diff > 0 else 'smoother'}.")

    lines.append("")
    lines.append("Report generated automatically by NeuroMotion AI.")

    return "\n".join(lines)


@app.post("/compare/automated_report")
async def compare_automated_report_endpoint(request: AutomatedReportRequest):
    """API endpoint wrapper for automated comparison report."""
    text = generate_automated_comparison_report(request.datasets)
    return {"report": text}


