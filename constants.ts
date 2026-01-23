
import { PipelineStage, StageConfig, ClinicalProfile, SkeletonFrame, Point3D } from './types';

export const STAGES: StageConfig[] = [
  { id: PipelineStage.INGESTION, label: "Ingestion", icon: "fa-video", description: "Video upload and preprocessing" },
  { id: PipelineStage.LIFTING_3D, label: "3D Lifting", icon: "fa-cube", description: "MediaPipe Holistic 3D pose extraction" },
  { id: PipelineStage.MOVEMENT_LAB, label: "Movement Lab", icon: "fa-flask", description: "Entropy, Fluency & Complexity analysis" },
  { id: PipelineStage.CLASSIFIER, label: "Diagnose", icon: "fa-user-md", description: "Hybrid Transformer-GCN Classification" },
  { id: PipelineStage.COMPLETE, label: "Results", icon: "fa-clipboard-check", description: "Clinical assessment report" }
];

// --- DETERMINISTIC RANDOM (PRNG) ---
let _seed = 123456;

export const setSeed = (s: number) => { 
    _seed = Math.abs(s) % 233280; 
};

const seededRandom = () => {
    _seed = (_seed * 9301 + 49297) % 233280;
    return _seed / 233280;
};

// --- RAW GEOMETRY GENERATORS (FULL BODY) ---

const getChaoticVal = (t: number) => Math.sin(t) + 0.5 * Math.sin(2.23 * t);
const getRhythmicVal = (t: number) => Math.sin(3 * t); // Consistent frequency
const getTremorVal = (t: number) => (seededRandom() - 0.5) * 0.8 + Math.sin(t * 15) * 0.2; // High freq jitter
const getSeizureVal = (t: number) => Math.sin(t * 15) * 0.8; // Fast rhythmic clonic jerking (High Amplitude)

// Helper to build a frame from a time t and a movement function
const buildFrame = (t: number, moveFn: (t: number) => number, type: 'CHAOTIC' | 'RHYTHMIC' | 'TREMOR' | 'FLACCID' | 'SEIZURE'): SkeletonFrame => {
    // Infant Anthropometry (Supine position, Top-down view)
    // Canvas is 0-100 x 0-100
    
    // Global Body Movement (Breathing/Writhing)
    const breath = type === 'FLACCID' ? Math.sin(t * 0.2) * 0.1 : Math.sin(t * 0.5) * 0.5;
    const writhing = type === 'FLACCID' ? 0 : Math.cos(t * 0.3) * 1.5;

    // Center point (with some writhing motion)
    const cx = 50 + (type === 'SEIZURE' ? 0 : writhing); // Seizures are rigid, less global writhing
    const neckY = 30 + breath; 
    
    // Proportions
    const torsoLen = 22;
    const hipY = neckY + torsoLen;
    const shoulderWidth = 14; 
    const pelvisWidth = 12;   
    const upperArm = 9;
    const forearm = 8;
    const thigh = 11;
    const shin = 9;

    // Helper to create point
    const pt = (x: number, y: number, z: number, stability: 'HIGH' | 'MED' | 'LOW'): Point3D => {
        let vBase = 0.98;
        if (stability === 'MED') vBase = 0.90;
        if (stability === 'LOW') vBase = 0.85; 
        const v = Math.min(1.0, Math.max(0.4, vBase + (seededRandom() * 0.1 - 0.05)));
        return { x, y, z, visibility: v };
    };

    // --- HEAD & EYES ---
    const rot = type === 'FLACCID' ? 0 : (type === 'SEIZURE' ? 0.02 : Math.sin(t * 0.2) * 0.05);
    const nose = pt(cx, 15 + breath, -20, 'HIGH'); 
    
    // Eye Deviation Logic
    let eyeOffsetX = 0;
    if (type === 'SEIZURE') {
        // Sustained lateral deviation (e.g., to the right)
        eyeOffsetX = 4.0; 
    } else if (type === 'CHAOTIC' || type === 'RHYTHMIC') {
        // Random looking around (Spontaneous Eye Opening)
        eyeOffsetX = Math.sin(t * 0.8) * 2;
    }
    
    const left_eye = pt(cx - 3 + eyeOffsetX, 12 + breath, -15, 'MED');
    const right_eye = pt(cx + 3 + eyeOffsetX, 12 + breath, -15, 'MED');

    // --- MOUTH (Crying Detection) ---
    // Simulate mouth movement relative to nose. 
    // Normal/Hyperalert: Active mouth (Crying/Sucking) -> High variability
    // Flaccid/Lethargic: Static
    let mouthOpen = 0;
    if (type === 'CHAOTIC' || type === 'TREMOR') {
        mouthOpen = Math.abs(Math.sin(t * 1.5)) * 1.5; // Frequent crying/vocalization
    } else if (type === 'FLACCID') {
        mouthOpen = 0.1; // Slightly open/slack
    } else {
        mouthOpen = 0.2; // Closed/Resting
    }
    const left_mouth = pt(cx - 1.5, 18 + breath + mouthOpen, -18, 'MED');
    const right_mouth = pt(cx + 1.5, 18 + breath + mouthOpen, -18, 'MED');

    // Shoulders
    const l_sh = pt(cx + shoulderWidth/2, neckY - (rot * 5), 0, 'HIGH');
    const r_sh = pt(cx - shoulderWidth/2, neckY + (rot * 5), 0, 'HIGH');
    
    // Hips
    const l_hip = pt(cx + pelvisWidth/2, hipY + (rot * 5), 0, 'HIGH');
    const r_hip = pt(cx - pelvisWidth/2, hipY - (rot * 5), 0, 'HIGH');

    // Kinematics Solver
    const solveLimb = (root: Point3D, len1: number, len2: number, phase: number, isLeg: boolean, side: 'L' | 'R') => {
        const sideMult = side === 'L' ? 1 : -1;
        
        let val1 = moveFn(t + phase);
        let val2 = (type === 'RHYTHMIC' || type === 'FLACCID' || type === 'SEIZURE') ? val1 : moveFn(t + phase + 1.5); 

        // Modifiers
        if (type === 'TREMOR') {
            val1 = val1 * 0.5; 
            val2 = val2 * 2.0; 
        }
        if (type === 'SEIZURE') {
            val1 = val1 * 0.8; // Controlled amplitude
            val2 = val2 * 0.8; 
        }

        // 1. Proximal Joint Rotation (Shoulder/Hip)
        let theta = 0; 
        let phi = 0;   

        if (isLeg) {
            if (type === 'FLACCID') {
                 theta = 0.8; phi = 0.8; 
            } else if (type === 'SEIZURE') {
                 // STIFFNESS: Rigid extension with rhythmic overlay
                 // Base angle 0.5 (Extension) + Rhythm
                 theta = 0.5 + (Math.abs(val1) * 0.2); 
                 phi = 0.3; // Adducted (Stiff)
            } else {
                 // NORMAL (Intermittent Flexion)
                 theta = 1.6 + (val1 * 0.4); 
                 phi = 0.4 + (val2 * 0.15); 
            }
        } else {
            if (type === 'FLACCID') {
                theta = 0.5; phi = 0.2;
            } else if (type === 'SEIZURE') {
                // STIFFNESS: Rigid flexion or extension ("Boxer" or decerebrate)
                // Let's simulate rigid flexion (Hypertonic)
                theta = 2.0 + (Math.abs(val1) * 0.2); 
                phi = 0.5; // Adducted
            } else {
                // NORMAL
                theta = 2.4 + (val1 * 0.35); 
                phi = 0.7 + (val2 * 0.25);
            }
        }

        const midX = root.x + (Math.sin(phi) * len1 * sideMult);
        const midY = root.y + (Math.cos(theta) * len1);
        const midZ = Math.abs(Math.sin(val1)) * 15; 
        const mid = pt(midX, midY, midZ, 'MED');

        // 2. Distal Joint Rotation
        let flex = 0;
        if (isLeg) {
            if (type === 'SEIZURE') flex = 0.3 + (val2 * 0.1); // Stiff Knee
            else flex = type === 'FLACCID' ? 0.2 : 1.5 + (val2 * 0.3); 
        } else {
            if (type === 'SEIZURE') flex = -2.0 + (val2 * 0.1); // Stiff Elbow
            else flex = type === 'FLACCID' ? 0.2 : -1.8 + (val2 * 0.3); 
        }
        
        const angle2 = theta + flex;

        const endX = mid.x + (Math.sin(phi * 0.8) * len1 * sideMult); 
        const endY = mid.y + (Math.cos(angle2) * len2);
        const endZ = Math.abs(Math.sin(val2)) * 10;

        const end = pt(endX, endY, endZ, 'LOW'); 

        return { mid, end };
    };

    const la = solveLimb(l_sh, upperArm, forearm, 0, false, 'L');
    const ra = solveLimb(r_sh, upperArm, forearm, (type==='RHYTHMIC' || type==='SEIZURE')?0:2.1, false, 'R');
    const ll = solveLimb(l_hip, thigh, shin, (type==='RHYTHMIC' || type==='SEIZURE')?0:1.3, true, 'L');
    const rl = solveLimb(r_hip, thigh, shin, (type==='RHYTHMIC' || type==='SEIZURE')?0:3.5, true, 'R');

    return {
        timestamp: t,
        joints: {
            nose, left_eye, right_eye, left_mouth, right_mouth,
            left_shoulder: l_sh, right_shoulder: r_sh,
            left_hip: l_hip, right_hip: r_hip,
            left_elbow: la.mid, left_wrist: la.end,
            right_elbow: ra.mid, right_wrist: ra.end,
            left_knee: ll.mid, left_ankle: ll.end,
            right_knee: rl.mid, right_ankle: rl.end
        }
    };
};

const generateNormalContinuousTraj = (frames: number): SkeletonFrame[] => {
  const data: SkeletonFrame[] = [];
  for (let i = 0; i < frames; i++) data.push(buildFrame(i * 0.1, getChaoticVal, 'CHAOTIC'));
  return data;
};

const generateIntermittentTraj = (frames: number): SkeletonFrame[] => {
  const data: SkeletonFrame[] = [];
  let t = 0;
  for (let i = 0; i < frames; i++) {
    const isPause = Math.sin(i * 0.05) > 0.5;
    if (!isPause) t += 0.1;
    data.push(buildFrame(t, getChaoticVal, 'CHAOTIC'));
  }
  return data;
};

const generateHyperalertTraj = (frames: number): SkeletonFrame[] => {
  const data: SkeletonFrame[] = [];
  for (let i = 0; i < frames; i++) data.push(buildFrame(i * 0.1, getTremorVal, 'TREMOR'));
  return data;
};

const generateFlaccidTraj = (frames: number): SkeletonFrame[] => {
  const data: SkeletonFrame[] = [];
  for (let i = 0; i < frames; i++) data.push(buildFrame(i * 0.02, getChaoticVal, 'FLACCID'));
  return data;
};

const generateSeizureTraj = (frames: number): SkeletonFrame[] => {
  const data: SkeletonFrame[] = [];
  for (let i = 0; i < frames; i++) {
    // Seizure moves faster than normal writhing
    data.push(buildFrame(i * 0.1, getSeizureVal, 'SEIZURE'));
  }
  return data;
};

export const CLINICAL_PROFILES: ClinicalProfile[] = [
  {
    id: 'NORMAL',
    label: 'Normal',
    features: {
      entropyScore: "High",
      fluencySAL: "Moderate",
      fractalDimension: "High",
      convexHullVolume: "Large",
      phaseSpaceTopology: "Cloud",
      clinicalNote: "Appropriate level of consciousness, normal tone (intermittent flexion), fluid movements. Crying and Spontaneous Eye Opening present."
    },
    trajectoryGenerator: generateNormalContinuousTraj
  },
  {
    id: 'SARNAT_I',
    label: 'Sarnat Stage I',
    features: {
      entropyScore: "Very High",
      fluencySAL: "Very Low (Tremors)",
      fractalDimension: "High (Noise)",
      convexHullVolume: "Restricted",
      phaseSpaceTopology: "Dense Cloud",
      clinicalNote: "Hyperalert, jittery, exaggerated reflexes, no seizures. Eyes wide open."
    },
    trajectoryGenerator: generateHyperalertTraj
  },
  {
    id: 'SARNAT_II',
    label: 'Sarnat Stage II',
    features: {
      entropyScore: "Variable",
      fluencySAL: "Low",
      fractalDimension: "Moderate",
      convexHullVolume: "Reduced",
      phaseSpaceTopology: "Cluster",
      clinicalNote: "Lethargic, hypotonic (extended), decreased activity with intermittent bursts."
    },
    trajectoryGenerator: generateIntermittentTraj
  },
  {
    id: 'SARNAT_III',
    label: 'Sarnat Stage III',
    features: {
      entropyScore: "Near Zero",
      fluencySAL: "Zero",
      fractalDimension: "Flatline",
      convexHullVolume: "Collapsed",
      phaseSpaceTopology: "Point",
      clinicalNote: "Stupor/Coma, flaccid tone, absent reflexes. No eye opening."
    },
    trajectoryGenerator: generateFlaccidTraj
  },
  {
    id: 'SEIZURE',
    label: 'Seizures',
    features: {
      entropyScore: "Moderate/Periodic",
      fluencySAL: "High Jerk + Stiffness",
      fractalDimension: "Low",
      convexHullVolume: "Moderate",
      phaseSpaceTopology: "Limit Cycle",
      clinicalNote: "Rhythmic clonic jerking with total body stiffness and sustained eye deviation."
    },
    trajectoryGenerator: generateSeizureTraj
  }
];

export const getProfileBySeed = (seed: number): ClinicalProfile => {
  // Deterministic profile selection based on seed
  // Use a localized PRNG calc so we don't mess up the global stream just for selection if we don't want to
  const localVal = (seed * 9301 + 49297) % 233280;
  const normalized = localVal / 233280;
  
  if (normalized < 0.25) return CLINICAL_PROFILES[0]; 
  if (normalized < 0.45) return CLINICAL_PROFILES[1]; 
  if (normalized < 0.65) return CLINICAL_PROFILES[2]; 
  if (normalized < 0.85) return CLINICAL_PROFILES[3]; 
  return CLINICAL_PROFILES[4]; 
};

export const getRandomProfile = (): ClinicalProfile => {
  // For LIVE mode or fallback, we can still use Random
  // But let's use seededRandom if we want consistency in session, or Math.random if truly random
  const rand = Math.random();
  if (rand < 0.25) return CLINICAL_PROFILES[0]; 
  if (rand < 0.45) return CLINICAL_PROFILES[1]; 
  if (rand < 0.65) return CLINICAL_PROFILES[2]; 
  if (rand < 0.85) return CLINICAL_PROFILES[3]; 
  return CLINICAL_PROFILES[4]; 
};
