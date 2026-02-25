"""
Shared Pydantic models for the Neuromotion AI backend.
Converted from types.ts — all TypeScript interfaces become Pydantic BaseModels.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Literal
from enum import IntEnum
import math


# --- Enums ---

class PipelineStage(IntEnum):
    IDLE = -1
    INGESTION = 0
    LIFTING_3D = 1
    MOVEMENT_LAB = 2
    CLASSIFIER = 3
    COMPLETE = 4


FMCategory = Literal[
    "Normal",
    "Sarnat Stage I",
    "Sarnat Stage II",
    "Sarnat Stage III",
    "Seizures",
]

SeizureType = Literal["None", "Clonic", "Myoclonic", "Tonic"]


# --- Core Data Models ---

class Point3D(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None  # Confidence 0.0-1.0


class SkeletonJoints(BaseModel):
    # Head & Face
    nose: Optional[Point3D] = None
    left_eye: Optional[Point3D] = None
    right_eye: Optional[Point3D] = None
    left_ear: Optional[Point3D] = None
    right_ear: Optional[Point3D] = None
    left_mouth: Optional[Point3D] = None
    right_mouth: Optional[Point3D] = None
    # Upper Body
    left_shoulder: Point3D
    right_shoulder: Point3D
    left_elbow: Point3D
    right_elbow: Point3D
    left_wrist: Point3D
    right_wrist: Point3D
    # Lower Body
    left_hip: Point3D
    right_hip: Point3D
    left_knee: Point3D
    right_knee: Point3D
    left_ankle: Point3D
    right_ankle: Point3D


class SkeletonFrame(BaseModel):
    timestamp: float
    joints: SkeletonJoints


class MovementMetrics(BaseModel):
    timestamp: float
    entropy: float = 0.0
    fluency_velocity: float = 0.0
    fluency_jerk: float = 0.0
    fractal_dim: float = 0.0
    phase_x: float = 0.0  # Position
    phase_v: float = 0.0  # Velocity
    kinetic_energy: float = 0.0
    angular_jerk: float = 0.0
    root_stress: float = 0.0
    bilateral_symmetry: float = 0.0
    lower_limb_kinetic_energy: float = 0.0
    com_velocity: float = 0.0
    head_stability: float = 0.0
    avg_visibility: float = 1.0  # Mean pose detection confidence across tracked joints
    min_visibility: float = 1.0  # Minimum joint confidence in this frame


class PostureMetrics(BaseModel):
    shoulder_flexion_index: float = 0.0
    hip_flexion_index: float = 0.0
    symmetry_score: float = 1.0  # 0 (Asymmetric) to 1 (Symmetric)
    tone_label: Literal["Hypotonic", "Normal", "Hypertonic"] = "Normal"
    frog_leg_score: float = 0.0
    spontaneous_activity: float = 0.0
    sustained_posture_score: float = 0.0
    crying_index: float = 0.0
    eye_openness_index: float = 0.0
    arousal_index: float = 0.0
    state_transition_probability: float = 0.0


class SeizureMetrics(BaseModel):
    rhythmicity_score: float = 0.0
    stiffness_score: float = 0.0
    eye_deviation_score: float = 0.0
    dominant_frequency: float = 0.0
    limb_synchrony: float = 0.0
    calculated_type: Literal["None", "Clonic", "Tonic", "Myoclonic"] = "None"


class MotionConfig(BaseModel):
    sensitivity: float = 0.85
    windowSize: int = 30
    entropyThreshold: float = 0.4
    jerkThreshold: float = 5.0
    rhythmicityWeight: float = 0.7
    stiffnessThreshold: float = 0.6


class ExpertCorrection(BaseModel):
    correctClassification: str  # FMCategory
    notes: str
    timestamp: str
    clinicianName: str


class RawData(BaseModel):
    entropy: float = 0.0
    fluency: float = 0.0
    complexity: float = 0.0
    variabilityIndex: float = 0.0
    csRiskScore: float = 0.0
    posture: PostureMetrics = PostureMetrics()
    seizure: SeizureMetrics = SeizureMetrics()
    avg_kinetic_energy: float = 0.0
    avg_root_stress: float = 0.0
    avg_bilateral_symmetry: float = 0.0
    avg_lower_limb_ke: float = 0.0
    avg_angular_jerk: float = 0.0
    avg_head_stability: float = 0.0
    avg_com_velocity: float = 0.0


class AnalysisReport(BaseModel):
    classification: str = "Normal"  # FMCategory
    confidence: float = 0.0
    seizureDetected: bool = False
    seizureType: str = "None"  # SeizureType
    differentialAlert: Optional[str] = None
    rawData: RawData = RawData()
    clinicalAnalysis: str = ""
    recommendations: List[str] = []
    timelineData: Optional[List[Dict[str, Any]]] = None
    expertCorrection: Optional[ExpertCorrection] = None


class SavedReport(AnalysisReport):
    id: str
    date: str
    videoName: str


class User(BaseModel):
    id: str
    name: str
    email: str


class UserWithPassword(User):
    password: str


# --- Request/Response Models ---

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class SaveReportRequest(BaseModel):
    report: AnalysisReport
    videoName: str


class CorrectionRequest(BaseModel):
    correction: ExpertCorrection


class TrajectoryRequest(BaseModel):
    profile_id: str
    frames: int = 300
    seed: Optional[int] = None


class CompareAIReportRequest(BaseModel):
    dataset_summaries: str  # Pre-formatted summary of datasets


class CompareChatRequest(BaseModel):
    question: str
    dataset_summaries: str


class CompareStatsRequest(BaseModel):
    data: List[Dict[str, Any]]


class AutomatedReportRequest(BaseModel):
    datasets: List[Dict[str, Any]]  # [{label, stats: {metric: {mean, min, max, std}}}]


class StageConfig(BaseModel):
    id: int  # PipelineStage value
    label: str
    icon: str
    description: str


class ClinicalProfileFeatures(BaseModel):
    entropyScore: str
    fluencySAL: str
    fractalDimension: str
    convexHullVolume: str
    phaseSpaceTopology: str
    clinicalNote: str


class ClinicalProfileInfo(BaseModel):
    """Clinical profile metadata (without the generator function)."""
    id: str
    label: str
    features: ClinicalProfileFeatures


# --- Request/Response Models ---

class AnalysisRequest(BaseModel):
    frames: List[Dict[str, Any]]
    config: MotionConfig


class RefineConfigRequest(BaseModel):
    current_report: Optional[Dict[str, Any]] = None
    expert_diagnosis: str
    annotation: str = ""
    current_config: Dict[str, float]


class ValidationRequest(BaseModel):
    timestamp: str
    ground_truth_classification: str
    doctor_notes: Optional[str] = None


class CaseSearchRequest(BaseModel):
    query: Optional[str] = None
    classifications: Optional[List[str]] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    biomarker_filters: Optional[Dict[str, Dict[str, float]]] = None
    validated_only: Optional[bool] = None
    similar_to: Optional[Dict[str, float]] = None
    top_k: int = 50
    method: str = "cosine"
