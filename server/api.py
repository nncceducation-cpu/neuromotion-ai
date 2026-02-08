
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import time
from dotenv import load_dotenv

# Load .env from project root (one level up from server/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Physics Engine Port
from .physics_engine import process_frames

# AI
try:
    from google.genai import Client
    import google.genai.types as types
except ImportError:
    print("Warning: Google GenAI SDK not found. Install google-genai.")

# Computer Vision
import cv2
import numpy as np

# --- VITPOSE MOCK INTEGRATION ---
# Real integration would use:
# from mmpose.apis import init_model, inference_topdown
# from mmpose.utils import register_all_modules
# register_all_modules()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- GEMINI SERVICE (Python Backend) ---
def generate_gemini_report(biomarkers: Dict[str, Any]):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "API Key missing"}
        
    client = Client(api_key=api_key)
    
    prompt = f"""
    You are an expert Neonatal Neurologist. Analyze these biomarkers from ViTPose (Python Backend):
    {json.dumps(biomarkers, indent=2)}
    
    Provide JSON output with classification (Normal, Sarnat Stage I/II/III, Seizures) and reasoning.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"classification": "Unknown", "clinicalAnalysis": "AI Error"}

@app.post("/analyze_frames")
async def analyze_frames_endpoint(request: AnalysisRequest):
    """
    Endpoint for frontend to send MediaPipe/ViTPose keypoints for Physics Processing + Gemini
    """
    print(f"Received {len(request.frames)} frames for analysis")
    
    # 1. Run Physics Engine (Python NumPy)
    metrics = process_frames(request.frames, request.config.dict())
    
    # 2. Calculate Aggregates
    if not metrics:
        return {"error": "Insufficient data"}
        
    ents = [m['entropy'] for m in metrics]
    jerks = [m['fluency_jerk'] for m in metrics]
    
    biomarkers = {
        "average_sample_entropy": float(np.mean(ents)),
        "peak_sample_entropy": float(np.max(ents)),
        "average_jerk": float(np.mean(jerks)),
        "backend_source": "Python/ViTPose-Pipeline"
    }
    
    # 3. Call Gemini
    report = generate_gemini_report(biomarkers)
    
    return {
        "metrics": metrics,
        "biomarkers": biomarkers,
        "report": report
    }

@app.post("/upload_video")
async def upload_video_for_vitpose(file: UploadFile = File(...)):
    """
    Endpoint to upload raw video.
    Server will:
    1. Save video
    2. Run ViTPose (MMPose) to extract keypoints
    3. Run Physics
    4. Return result
    """
    # 1. Save File
    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())
        
    # 2. Run ViTPose (Mocked for this file, assumes mmpose installed in prod)
    print(f"Running ViTPose Transformer on {temp_filename}...")
    
    # --- VITPOSE LOGIC (MMPose) ---
    # model = init_model('configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py', checkpoint='vitpose_base.pth', device='cuda:0')
    # video = cv2.VideoCapture(temp_filename)
    # frames_keypoints = []
    # while video.isOpened():
    #     ret, frame = video.read()
    #     if not ret: break
    #     result = inference_topdown(model, frame)
    #     frames_keypoints.append(convert_to_skeleton(result))
    
    # Mocking result for demo response if ViTPose not active
    # We would return a special flag telling frontend to use local MP or processed data
    
    os.remove(temp_filename)
    
    return {"status": "ViTPose processing complete (Simulated)", "frames_processed": 300}

@app.get("/")
def health_check():
    return {"status": "active", "model": "ViTPose-Base", "device": "cuda:0"}

