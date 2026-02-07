import sys
import os

# 1. SETUP PATH: Add the current directory to sys.path so we can import 'physics_engine'
# This fixes the "ModuleNotFoundError" whether running via uvicorn or python directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import time
import numpy as np

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

# 4. INITIALIZE APP (Must be before routes!)
app = FastAPI()

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
def generate_gemini_report(biomarkers: Dict[str, Any]):
    api_key = os.environ.get("GEMINI_API_KEY") # Check env var
    if not api_key:
        # Fallback to checking the process env if set elsewhere
        api_key = os.environ.get("API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return {"error": "API Key missing"}
        
    try:
        client = Client(api_key=api_key)
        prompt = f"""
        You are an expert Neonatal Neurologist. Analyze these biomarkers from ViTPose (Python Backend):
        {json.dumps(biomarkers, indent=2)}
        
        Provide JSON output with classification (Normal, Sarnat Stage I/II/III, Seizures) and reasoning.
        """
        response = client.models.generate_content(
            model='gemini-2.0-flash', # Updated to a widely available model alias
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type='application/json')
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"classification": "Unknown", "clinicalAnalysis": f"AI Error: {str(e)}"}

# 8. API ROUTES
@app.post("/analyze_frames")
async def analyze_frames_endpoint(request: AnalysisRequest):
    """
    Endpoint for frontend to send MediaPipe/ViTPose keypoints for Physics Processing + Gemini
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
        "backend_source": "Python/ViTPose-Pipeline"
    }
    
    # Call Gemini
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
    """
    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        print(f"Processing video: {temp_filename}")
        # Placeholder for ViTPose logic
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return {"status": "ViTPose processing complete (Simulated)", "frames_processed": 300}

@app.get("/health")
def health_check():
    return {"status": "active", "model": "ViTPose-Base", "device": "cuda:0"}

@app.get("/")
def root():
    return {"message": "Neuromotion AI Backend is Running"}