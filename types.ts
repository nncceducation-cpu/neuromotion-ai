
export enum PipelineStage {
  IDLE = -1,
  INGESTION = 0,
  LIFTING_3D = 1,
  MOVEMENT_LAB = 2,
  CLASSIFIER = 3,
  COMPLETE = 4
}

export interface StageConfig {
  id: PipelineStage;
  label: string;
  icon: string;
  description: string;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
  visibility?: number; // Confidence score 0.0 - 1.0
}

export interface SkeletonFrame {
  timestamp: number;
  joints: {
    // Head & Face
    nose?: Point3D;
    left_eye?: Point3D;
    right_eye?: Point3D;
    left_mouth?: Point3D; // New: For Crying detection
    right_mouth?: Point3D; // New: For Crying detection
    
    // Upper Body
    left_shoulder: Point3D;
    right_shoulder: Point3D;
    left_elbow: Point3D;
    right_elbow: Point3D;
    left_wrist: Point3D;
    right_wrist: Point3D;
    
    // Lower Body
    left_hip: Point3D;
    right_hip: Point3D;
    left_knee: Point3D;
    right_knee: Point3D;
    left_ankle: Point3D;
    right_ankle: Point3D;
  };
}

export interface MovementMetrics {
  timestamp: number;
  // Metric 1: Entropy (Predictability)
  entropy: number; 
  // Metric 2: Fluency (Smoothness/SAL)
  fluency_velocity: number;
  fluency_jerk: number;
  // Metric 3: Complexity (Fractal Dimension)
  fractal_dim: number;
  // Metric 4: Variability (Phase Space)
  phase_x: number; // Position
  phase_v: number; // Velocity
  
  // --- NEW UNREAL ENGINE PHYSICS METRICS ---
  kinetic_energy: number; // Total body kinetic energy (Rigid Body Dynamics)
  angular_jerk: number; // Smoothness of joint rotations (Kinematics)
  root_stress: number; // Deviation of Root/CoM (Stability)
}

export interface PostureMetrics {
  shoulder_flexion_index: number; 
  hip_flexion_index: number;      
  symmetry_score: number; // 0 (Asymmetric) to 1 (Symmetric)
  tone_label: 'Hypotonic' | 'Normal' | 'Hypertonic';
  frog_leg_score: number; 
  spontaneous_activity: number;
  sustained_posture_score: number; // 0-1 (High = Lethargy/Coma, Low = Normal/Active)
  // New: Consciousness Biomarkers
  crying_index: number; // 0-1 (High = Crying/Vocalizing)
  eye_openness_index: number; // 0-1 (High = Spontaneous Open/Scanning)
  // New: HIE Lethargy / Arousal Biomarkers
  arousal_index: number; // Magnitude of entropy spike during activity burst
  state_transition_probability: number; // Probability of switching from Quiet -> Active
}

export interface SeizureMetrics {
  rhythmicity_score: number; // 0-1
  stiffness_score: number; // 0-1 (Low Variance)
  eye_deviation_score: number; // 0-1 (Sustained Deviation)
  dominant_frequency: number; // Hz (New: State of the Art Frequency Analysis)
  limb_synchrony: number; // 0-1 (New: Cross-limb correlation)
  calculated_type: 'None' | 'Clonic' | 'Tonic' | 'Myoclonic'; // Strict mathematical classification
}

export type FMCategory = 
  | "Normal"
  | "Sarnat Stage I"
  | "Sarnat Stage II"
  | "Sarnat Stage III"
  | "Seizures";

export type SeizureType = 
  | "None" 
  | "Clonic" 
  | "Myoclonic" 
  | "Tonic";

export interface ClinicalProfile {
  id: string;
  label: FMCategory;
  features: {
    entropyScore: string;
    fluencySAL: string;
    fractalDimension: string;
    convexHullVolume: string;
    phaseSpaceTopology: string;
    clinicalNote: string;
  };
  trajectoryGenerator: (frames: number) => SkeletonFrame[];
}

export interface ExpertCorrection {
  correctClassification: FMCategory;
  notes: string;
  timestamp: string;
  clinicianName: string;
}

export interface AnalysisReport {
  // Result Type 1: Categorical
  classification: FMCategory;
  confidence: number;
  
  // New Algorithms Fields from Vertex AI Integration
  seizureDetected: boolean;
  seizureType: SeizureType;
  differentialAlert?: string; // Mimic warnings (e.g., Jitteriness vs Seizure)

  // Result Type 2: Raw Data Summary
  rawData: {
    entropy: number;
    fluency: number;
    complexity: number;
    variabilityIndex: number;
    csRiskScore: number; // Autocorrelation Risk Score
    posture: PostureMetrics; // Posture Assessment
    seizure: SeizureMetrics; // New: Seizure specific biomarkers
    
    // New Physics Summary
    avg_kinetic_energy: number;
    avg_root_stress: number;
  };

  // Result Type 3: Clinical
  clinicalAnalysis: string;
  recommendations: string[];
  
  // Result Type 4: Full Time Series (For Export)
  timelineData?: MovementMetrics[];

  // Expert Feedback Loop
  expertCorrection?: ExpertCorrection;
}

export interface User {
  id: string;
  name: string;
  email: string;
}

export interface SavedReport extends AnalysisReport {
  id: string;
  date: string;
  videoName: string;
}

// --- NEW TYPES FOR ADVANCED TRAINING ---

export interface MotionConfig {
  sensitivity: number;
  windowSize: number;
  entropyThreshold: number;
  jerkThreshold: number;
  rhythmicityWeight: number;
  stiffnessThreshold: number;
}

export interface AggregatedStats {
  isEncephalopathy: boolean;
  dominantSarnatStage: string;
  avgEntropy: number;
  maxSeizureProb: number;
  windowDuration: number;
  interictalConsciousness: string;
}

// --- NEW TYPES FOR COMPARISON MODE ---
export interface ComparisonDataset {
  label: string;
  data: MovementMetrics[];
  stats?: Record<string, { mean: number; min: number; max: number; std: number }>;
}