import sys
import os

# 1. SETUP PATH: Add the current directory to sys.path so we can import 'physics_engine'
# This fixes the "ModuleNotFoundError" whether running via uvicorn or python directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone  # For timestamping logged data
from dotenv import load_dotenv

# Load .env from project root (one level up from server/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# 2. ROBUST IMPORT: Try importing physics_engine both ways (local vs module)
try:
    from physics_engine import process_frames
except ImportError:
    try:
        from .physics_engine import process_frames
    except ImportError:
        print("CRITICAL WARNING: 'physics_engine.py' not found. Ensure it is in the server/ directory.")
        process_frames = lambda frames, config: [] # Fallback to prevent crash

# 3. AI SDK IMPORT
try:
    from google.genai import Client
    import google.genai.types as types
except ImportError:
    print("Warning: Google GenAI SDK not found. Install google-genai.")
    Client = None  # type: ignore[assignment,misc]
    types = None  # type: ignore[assignment]

# 3b. YOLO26 POSE IMPORT (optional — graceful fallback if not installed)
try:
    from yolo_inference import load_models, process_video, is_loaded
    YOLO_AVAILABLE = True
except ImportError:
    print("Info: yolo_inference not available. YOLO26 Pose disabled, MediaPipe-only mode.")
    YOLO_AVAILABLE = False
    # Define stub functions for type checking
    def load_models() -> bool:
        return False
    def process_video(video_path: str, target_fps: float = 10.0) -> list:
        return []
    def is_loaded() -> bool:
        return False

# 4. INITIALIZE APP (Must be before routes!)
app = FastAPI()

# 4b. STARTUP: Load YOLO26 Pose model once
@app.on_event("startup")
async def startup_event():
    if YOLO_AVAILABLE:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, load_models)
        if success:
            print("YOLO26 Pose model loaded successfully at startup.")
        else:
            print("WARNING: YOLO26 Pose model failed to load. /upload_video will be unavailable.")
    else:
        print("YOLO26 Pose not installed. Running in MediaPipe-only mode.")

# GPU semaphore: prevent concurrent inference (not thread-safe)
_gpu_semaphore = asyncio.Semaphore(1)

# 5. CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. DATA MODELS
class MotionConfig(BaseModel):
    sensitivity: float
    windowSize: int
    entropyThreshold: float
    jerkThreshold: float
    rhythmicityWeight: float
    stiffnessThreshold: float

class AnalysisRequest(BaseModel):
    frames: List[Dict[str, Any]]
    config: MotionConfig

class AnalysisResponse(BaseModel):
    metrics: List[Dict[str, Any]]
    biomarkers: Dict[str, Any]
    report: Dict[str, Any]

# 7. HELPER FUNCTIONS

def log_analysis_result(biomarkers: Dict[str, Any], gemini_response: Dict[str, Any],
                        ground_truth: Optional[str] = None, metadata: Optional[Dict] = None,
                        first_frame_skeleton: Optional[Dict[str, Any]] = None):
    """
    Logs analysis results to a JSONL file for future model training.

    **Why we do this:** Every analysis is a potential training example. By logging:
    - biomarkers (input features)
    - gemini_response (AI prediction)
    - ground_truth (doctor's diagnosis, when available)
    - first_frame_skeleton (for visual verification of pose detection)
    We build a dataset that can be used to train a specialized fine-tuned model later.

    **JSONL format:** Each line is a separate JSON object. This is better than a single
    JSON array because it allows appending new records without reading the entire file.

    **NEW: Skeleton visualization** - We now log the first frame's skeleton with all
    17 COCO keypoints labeled. This lets doctors verify that YOLO26 detected the pose
    correctly before trusting the analysis.

    Args:
        biomarkers: The computed motion metrics (entropy, jerk, etc.)
        gemini_response: Gemini's classification and reasoning
        ground_truth: Optional doctor-verified diagnosis (add this via API later)
        metadata: Optional extra info (video_id, patient_age, etc.)
        first_frame_skeleton: First frame's keypoints for visual verification
    """
    log_dir = os.path.join(os.path.dirname(__file__), "analysis_logs")
    os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Use JSONL format: one JSON object per line
    # This is the standard format for ML training datasets
    log_file = os.path.join(log_dir, "gemini_predictions.jsonl")

    # Label skeleton keypoints with COCO names for clarity
    skeleton_with_labels = None
    if first_frame_skeleton and "joints" in first_frame_skeleton:
        # COCO 17 keypoint names (standard order)
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

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # ISO format: 2026-02-08T15:30:00.123456+00:00
        "biomarkers": biomarkers,
        "gemini_prediction": gemini_response,
        "ground_truth": ground_truth,  # null until doctor validates
        "doctor_notes": None,  # Will be filled via /validate endpoint
        "first_frame_skeleton": skeleton_with_labels,  # For visual verification
        "metadata": metadata or {}
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        print(f"Logged analysis to {log_file}")
    except Exception as e:
        print(f"Warning: Failed to log analysis: {e}")


def generate_gemini_report(biomarkers: Dict[str, Any]):
    """
    Sends biomarkers to Gemini 2.5 Flash for clinical classification.

    **Improvements made:**
    1. **Structured output schema:** We specify the exact JSON structure Gemini must return
    2. **Few-shot learning:** We provide 2 example cases to guide Gemini's reasoning
    3. **Confidence scoring:** We ask Gemini to include a confidence level (0.0-1.0)

    **What is few-shot learning?**
    Instead of just describing the task, we show the AI 2-3 examples of input -> output.
    This dramatically improves accuracy because the AI learns the pattern by example.
    Think of it like showing a student sample problems before giving them the real test.
    """
    api_key = os.environ.get("GEMINI_API_KEY") # Check env var
    if not api_key:
        # Fallback to checking the process env if set elsewhere
        api_key = os.environ.get("API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return {"error": "API Key missing"}

    if Client is None or types is None:
        print("Error: Google GenAI SDK not installed.")
        return {"error": "Google GenAI SDK not installed"}

    try:
        client = Client(api_key=api_key)

        # IMPROVEMENT #1 & #2: Structured schema + Few-shot examples
        # We provide example cases so Gemini learns the reasoning pattern
        prompt = f"""You are an expert Neonatal Neurologist analyzing infant motion biomarkers for signs of neurological impairment.

**BACKGROUND:**
- Sample Entropy: Measures movement complexity (0.3-0.6 = normal, >0.7 = dysregulated/chaotic, <0.2 = overly rigid)
- Jerk (fluency): Measures smoothness (4-7 = normal, >8 = jerky/uncoordinated, <3 = hypokinetic)
- Typical ranges based on research: Normal infants show entropy ~0.4+-0.15, jerk ~5.5+-1.5

**FEW-SHOT EXAMPLES (learn from these):**

Example 1:
Input: {{"average_sample_entropy": 0.38, "peak_sample_entropy": 0.52, "average_jerk": 5.2}}
Output: {{
  "classification": "Normal",
  "confidence": 0.92,
  "reasoning": "All biomarkers within normal ranges. Entropy of 0.38 indicates healthy movement variability. Jerk of 5.2 suggests smooth, coordinated motion typical of neurologically intact infants."
}}

Example 2:
Input: {{"average_sample_entropy": 0.82, "peak_sample_entropy": 1.15, "average_jerk": 9.3}}
Output: {{
  "classification": "Sarnat Stage II",
  "confidence": 0.78,
  "reasoning": "Elevated entropy (0.82, >1.5 SD above normal) indicates dysregulated, chaotic movements. High jerk (9.3) suggests impaired motor control. Pattern consistent with moderate HIE. Recommend continuous monitoring and EEG correlation."
}}

**NOW ANALYZE THIS CASE:**
Input: {json.dumps(biomarkers, indent=2)}

**REQUIRED OUTPUT FORMAT (strict JSON schema):**
{{
  "classification": "Normal" | "Sarnat Stage I" | "Sarnat Stage II" | "Sarnat Stage III" | "Seizures" | "Uncertain",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<2-3 sentence clinical explanation citing specific biomarker values>",
  "recommendations": "<optional: suggest EEG, imaging, or clinical actions if abnormal>"
}}

**IMPORTANT:**
- Use "Uncertain" if biomarkers are ambiguous or contradictory
- Confidence should reflect how clear-cut the pattern is
- Always cite specific values in your reasoning (e.g., "entropy of 0.82")
"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )

        # Parse response and ensure it has the required fields
        if response.text is None:
            return {"classification": "Error", "confidence": 0.0, "reasoning": "Gemini returned empty response"}
        result = json.loads(response.text)

        # Validate schema (basic sanity check)
        required_fields = ["classification", "confidence", "reasoning"]
        for field in required_fields:
            if field not in result:
                print(f"Warning: Gemini response missing '{field}' field")
                result[field] = "Unknown" if field == "classification" else 0.0 if field == "confidence" else "No reasoning provided"

        return result

    except Exception as e:
        print(f"Gemini Error: {e}")
        return {
            "classification": "Error",
            "confidence": 0.0,
            "reasoning": f"AI service error: {str(e)}",
            "recommendations": "Manual review required"
        }

# 8. API ROUTES
@app.post("/analyze_frames")
async def analyze_frames_endpoint(request: AnalysisRequest):
    """
    Endpoint for frontend to send MediaPipe/YOLO26 keypoints for Physics Processing + Gemini
    """
    print(f"Received {len(request.frames)} frames for analysis")

    # Run Physics Engine
    metrics = process_frames(request.frames, request.config.model_dump())

    # Handle empty metrics gracefully
    if not metrics:
        print("Physics engine returned no metrics.")
        return {
            "metrics": [],
            "biomarkers": {"error": "Insufficient motion data extracted"},
            "report": {}
        }

    # Extract aggregates safely
    ents = [m.get('entropy', 0) for m in metrics]
    jerks = [m.get('fluency_jerk', 0) for m in metrics]

    if not ents:
        return {"error": "No entropy data", "metrics": metrics}

    biomarkers = {
        "average_sample_entropy": float(np.mean(ents)),
        "peak_sample_entropy": float(np.max(ents)),
        "average_jerk": float(np.mean(jerks)) if jerks else 0.0,
        "backend_source": "Python/YOLO26-Pipeline"
    }

    # Call Gemini (few-shot + structured output)
    report = generate_gemini_report(biomarkers)

    # Log everything for future model training
    log_analysis_result(
        biomarkers=biomarkers,
        gemini_response=report,
        ground_truth=None,
        metadata={"source": "frontend_mediapipe", "frame_count": len(request.frames)},
        first_frame_skeleton=request.frames[0] if request.frames else None
    )

    return {
        "metrics": metrics,
        "biomarkers": biomarkers,
        "report": report
    }

@app.post("/upload_video")
async def upload_video_for_pose(file: UploadFile = File(...)):
    """
    Upload a video for server-side YOLO26 Pose processing.
    Runs: detection + pose estimation -> physics engine -> Gemini report.
    """
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

        # Run YOLO26 pipeline (GPU-exclusive via semaphore)
        async with _gpu_semaphore:
            loop = asyncio.get_event_loop()
            skeleton_frames = await loop.run_in_executor(
                None, lambda: process_video(temp_filename, target_fps=10.0)
            )

        if len(skeleton_frames) < 10:
            return {
                "metrics": [], "biomarkers": {"error": "Insufficient pose data from video"},
                "report": {"error": "Too few valid frames"}, "frames_processed": len(skeleton_frames)
            }

        # Reuse existing physics engine
        metrics = process_frames(skeleton_frames, config.model_dump())
        if not metrics:
            return {
                "metrics": [], "biomarkers": {"error": "Physics engine returned no metrics"},
                "report": {}, "frames_processed": len(skeleton_frames)
            }

        ents = [m.get('entropy', 0) for m in metrics]
        jerks = [m.get('fluency_jerk', 0) for m in metrics]
        biomarkers = {
            "average_sample_entropy": float(np.mean(ents)),
            "peak_sample_entropy": float(np.max(ents)),
            "average_jerk": float(np.mean(jerks)) if jerks else 0.0,
            "backend_source": "Python/YOLO26-Pipeline",
            "frames_processed": len(skeleton_frames)
        }

        report = generate_gemini_report(biomarkers)

        # Log this analysis for future training
        log_analysis_result(
            biomarkers=biomarkers,
            gemini_response=report,
            ground_truth=None,
            metadata={
                "source": "video_upload",
                "filename": file.filename,
                "frames_processed": len(skeleton_frames)
            },
            first_frame_skeleton=skeleton_frames[0] if skeleton_frames else None
        )

        return {
            "metrics": metrics, "biomarkers": biomarkers,
            "report": report, "frames_processed": len(skeleton_frames)
        }
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
    return {"message": "Neuromotion AI Backend is Running"}

# Doctor Validation Endpoint for Ground Truth Labels
class ValidationRequest(BaseModel):
    """
    Data model for doctor validation of AI predictions.

    This allows clinicians to correct AI mistakes, which builds a labeled
    dataset for training a fine-tuned model in the future.
    """
    timestamp: str  # ISO timestamp of the original analysis (from log file)
    ground_truth_classification: str  # Doctor's actual diagnosis
    doctor_notes: Optional[str] = None  # Optional clinical context

@app.post("/validate")
async def validate_prediction(request: ValidationRequest):
    """
    Endpoint for doctors to submit ground truth labels for AI predictions.

    **Use case:**
    After a patient is diagnosed, a neurologist can use this endpoint to
    label the AI's prediction as correct/incorrect and provide the true diagnosis.

    **How it works:**
    1. Find the log entry with matching timestamp
    2. Update the ground_truth field with doctor's diagnosis
    3. This labeled data can later be used to train a specialized model
    """
    log_dir = os.path.join(os.path.dirname(__file__), "analysis_logs")
    log_file = os.path.join(log_dir, "gemini_predictions.jsonl")

    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail="No analysis logs found")

    try:
        # Read all log entries
        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))

        # Find matching entry by timestamp
        updated = False
        for entry in entries:
            if entry["timestamp"] == request.timestamp:
                entry["ground_truth"] = request.ground_truth_classification
                entry["doctor_notes"] = request.doctor_notes
                entry["validated_at"] = datetime.now(timezone.utc).isoformat()
                updated = True
                break

        if not updated:
            raise HTTPException(status_code=404, detail=f"No analysis found with timestamp {request.timestamp}")

        # Write back to file (rewrite entire file with updated data)
        with open(log_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return {
            "status": "success",
            "message": f"Ground truth label '{request.ground_truth_classification}' saved for analysis at {request.timestamp}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/pending_validations")
async def get_pending_validations(limit: int = 50, include_validated: bool = False):
    """
    Fetch analysis cases that need doctor validation.

    **Query params:**
    - limit: Max number of cases to return (default 50)
    - include_validated: If True, return all cases; if False, only unvalidated (default False)

    **Returns:**
    List of analyses sorted by timestamp (newest first), each containing:
    - timestamp, biomarkers, gemini_prediction, first_frame_skeleton, metadata
    - ground_truth (if validated)
    - doctor_notes (if validated)
    """
    log_dir = os.path.join(os.path.dirname(__file__), "analysis_logs")
    log_file = os.path.join(log_dir, "gemini_predictions.jsonl")

    if not os.path.exists(log_file):
        return {"cases": [], "total": 0}

    try:
        # Read all log entries
        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))

        # Filter based on validation status
        if not include_validated:
            entries = [e for e in entries if not e.get("ground_truth")]

        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x["timestamp"], reverse=True)

        # Limit results
        entries = entries[:limit]

        return {
            "cases": entries,
            "total": len(entries),
            "showing": "all" if include_validated else "unvalidated_only"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch validations: {str(e)}")
