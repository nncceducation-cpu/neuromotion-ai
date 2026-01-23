
import { Point3D, SkeletonFrame, MovementMetrics, PostureMetrics, SeizureMetrics, MotionConfig } from '../types';

// --- UNREAL ENGINE MATH LIBRARY (UE5 Inspired) ---
const UEMath = {
    // FVector Dot Product
    dot: (v1: {x:number, y:number, z:number}, v2: {x:number, y:number, z:number}) => 
        v1.x*v2.x + v1.y*v2.y + v1.z*v2.z,
        
    // FVector Cross Product
    cross: (v1: {x:number, y:number, z:number}, v2: {x:number, y:number, z:number}) => ({
        x: v1.y*v2.z - v1.z*v2.y,
        y: v1.z*v2.x - v1.x*v2.z,
        z: v1.x*v2.y - v1.y*v2.x
    }),
    
    // FVector Size/Magnitude
    size: (v: {x:number, y:number, z:number}) => 
        Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z),
        
    // FVector Dist
    dist: (p1: Point3D, p2: Point3D) => 
        Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2) + Math.pow(p2.z - p1.z, 2)),

    // FVector Normalize
    normalize: (v: {x:number, y:number, z:number}) => {
        const mag = Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        return mag === 0 ? {x:0, y:0, z:0} : {x: v.x/mag, y: v.y/mag, z: v.z/mag};
    },

    // Create vector from two points
    vector: (from: Point3D, to: Point3D) => ({
        x: to.x - from.x,
        y: to.y - from.y,
        z: to.z - from.z
    }),

    // Angle between two vectors (in degrees)
    angle: (v1: {x:number, y:number, z:number}, v2: {x:number, y:number, z:number}) => {
        const mag1 = Math.sqrt(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
        const mag2 = Math.sqrt(v2.x*v2.x + v2.y*v2.y + v2.z*v2.z);
        if (mag1 === 0 || mag2 === 0) return 0;
        const dot = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
        return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2)))) * (180 / Math.PI);
    }
};

// --- ADVANCED SIGNAL PROCESSING (State of the Art) ---

// Simple Moving Average (SMA) for noise reduction/smoothing
const smoothSignal = (data: number[], window: number = 3): number[] => {
    if (data.length < window) return data;
    const result: number[] = [];
    for(let i = 0; i < data.length; i++) {
        let sum = 0;
        let count = 0;
        const start = Math.max(0, i - Math.floor(window/2));
        const end = Math.min(data.length - 1, i + Math.floor(window/2));
        for(let j = start; j <= end; j++) {
            sum += data[j];
            count++;
        }
        result.push(sum / count);
    }
    return result;
};

// Recurrence Rate (RR) - Measures the probability that a state recurs
// High RR = Periodic/Seizure; Low RR = Chaotic/Normal
const calculateRecurrenceRate = (data: number[], epsilon: number): number => {
    if(data.length < 5) return 0;
    let recurrencePoints = 0;
    const N = data.length;
    // Downsample for performance if needed, but for windowSize ~30-60 it's fine O(N^2)
    for(let i = 0; i < N; i++) {
        for(let j = 0; j < N; j++) {
            if(Math.abs(data[i] - data[j]) < epsilon) {
                recurrencePoints++;
            }
        }
    }
    // Normalized by N^2
    return recurrencePoints / (N * N);
};

// Calculate Sample Entropy (SampEn) - Robust Implementation
const calculateSampleEntropy = (data: number[], m: number = 2, r: number = 0.2): number => {
  if (data.length < 10) return 0;
  
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const variance = data.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / data.length;
  const std = Math.sqrt(variance);
  
  // Normalize signal to unit variance (State of the Art practice for Entropy)
  const normalized = std > 0 ? data.map(val => (val - mean) / std) : data;
  const threshold = r; // Since we normalized, r is effectively r*std

  const getCount = (len: number) => {
    let count = 0;
    const N = normalized.length;
    for (let i = 0; i < N - len; i++) {
      const template = normalized.slice(i, i + len);
      for (let j = 0; j < N - len; j++) {
        if (i === j) continue;
        const segment = normalized.slice(j, j + len);
        // Chebyshev distance (Infinity Norm)
        let maxDist = 0;
        for(let k=0; k<len; k++) {
            maxDist = Math.max(maxDist, Math.abs(template[k] - segment[k]));
        }
        if (maxDist < threshold) count++;
      }
    }
    return count;
  };

  const A = getCount(m + 1);
  const B = getCount(m);

  if (B === 0) return 0; // Avoid division by zero
  if (A === 0) return -Math.log(1 / B); // Approx for zero match
  return -Math.log(A / B);
};

// Higuchi Fractal Dimension (HFD) - Efficient calculation
const calculateFractalDimension = (data: number[], kMax: number = 5): number => {
  if (data.length < 10) return 1.5; // Default neutral complexity
  const N = data.length;
  const Lk: number[] = [];
  
  for (let k = 1; k <= kMax; k++) {
    let Lm = 0;
    for (let m = 0; m < k; m++) {
      let Lmk = 0;
      const maxIdx = Math.floor((N - m - 1) / k);
      for (let i = 1; i <= maxIdx; i++) {
        Lmk += Math.abs(data[m + i * k] - data[m + (i - 1) * k]);
      }
      const norm = (N - 1) / (maxIdx * k);
      Lmk = Lmk * norm / k;
      Lm += Lmk;
    }
    Lk.push(Lm / k);
  }
  
  // Linear regression of ln(L(k)) vs ln(1/k)
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  const nPoints = Lk.length;
  for (let k = 1; k <= nPoints; k++) {
      const x = Math.log(1/k);
      const y = Math.log(Lk[k-1]);
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumXX += x * x;
  }
  
  const slope = (nPoints * sumXY - sumX * sumY) / (nPoints * sumXX - sumX * sumX);
  return slope; // HFD is the slope
};

// --- NEW GMA SPECIFIC ALGORITHMS ---

// Pearson Correlation Coefficient (More robust than Cosine for time-series)
const calculatePearsonCorrelation = (x: number[], y: number[]): number => {
    if (x.length !== y.length || x.length === 0) return 0;
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = (n * sumXY) - (sumX * sumY);
    const denominator = Math.sqrt(((n * sumX2) - sumX ** 2) * ((n * sumY2) - sumY ** 2));

    if (denominator === 0) return 0;
    return numerator / denominator;
};

const calculateCosineSimilarity = (anchor: Point3D, endA: Point3D, endB: Point3D): number => {
    const vecA = UEMath.vector(anchor, endA);
    const vecB = UEMath.vector(anchor, endB);

    const dot = UEMath.dot(vecA, vecB);
    const magA = UEMath.size(vecA);
    const magB = UEMath.size(vecB);

    if (magA === 0 || magB === 0) return 0;
    return Math.max(-1, Math.min(1, dot / (magA * magB)));
};

// Discrete Fourier Transform (DFT) for Frequency Extraction
const calculateFrequencyAnalysis = (data: number[], fps: number = 30): { dominantFreq: number, powerRatio: number } => {
    const N = data.length;
    if (N < 5) return { dominantFreq: 0, powerRatio: 0 };

    let maxPower = 0;
    let dominantFreq = 0;
    let totalPower = 0;
    
    // Analyze 0.5Hz to 15Hz
    for (let k = 1; k < N / 2; k++) {
        let real = 0;
        let imag = 0;
        for (let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            real += data[n] * Math.cos(angle);
            imag -= data[n] * Math.sin(angle);
        }
        const power = Math.sqrt(real*real + imag*imag);
        totalPower += power;

        if (power > maxPower) {
            maxPower = power;
            dominantFreq = (k * fps) / N;
        }
    }
    
    const powerRatio = totalPower > 0 ? maxPower / totalPower : 0;
    return { dominantFreq, powerRatio };
};

// Peak Interval Regularity (Time Domain)
const calculatePeakRegularity = (data: number[]): number => {
    const peaks: number[] = [];
    const meanVal = data.reduce((a,b)=>a+b,0)/data.length;
    
    for(let i=1; i<data.length-1; i++) {
        if (data[i] > data[i-1] && data[i] > data[i+1]) {
            if (data[i] > meanVal * 1.1) {
                peaks.push(i);
            }
        }
    }

    if (peaks.length < 3) return 0;

    const intervals: number[] = [];
    for(let i=1; i<peaks.length; i++) {
        intervals.push(peaks[i] - peaks[i-1]);
    }

    const meanInt = intervals.reduce((a,b)=>a+b,0)/intervals.length;
    const varInt = intervals.reduce((a,b)=>a+Math.pow(b-meanInt,2),0)/intervals.length;
    const stdInt = Math.sqrt(varInt);
    
    if (meanInt === 0) return 0;
    const cv = stdInt / meanInt; 
    const score = Math.max(0, 1 - (cv / 0.6));
    return score;
};

// --- CORE PHYSICS ENGINE ---

export const physicsEngine = {
  
  processSignal: (frames: SkeletonFrame[], config?: MotionConfig): MovementMetrics[] => {
    const metrics: MovementMetrics[] = [];
    
    // Dynamic Configuration
    const windowSize = config?.windowSize ?? 30;
    const entropyR = config?.entropyThreshold ?? 0.2; 
    const jerkOffset = config?.jerkThreshold ?? 5.0;  

    // 1. SIGNAL SMOOTHING
    const rawX = frames.map(f => f.joints.right_wrist.x);
    const rawY = frames.map(f => f.joints.right_wrist.y);
    const smoothX = smoothSignal(rawX, 3);
    const smoothY = smoothSignal(rawY, 3);

    // Root (CoM approximation using Hip Center)
    const rootX = frames.map(f => (f.joints.left_hip.x + f.joints.right_hip.x) / 2);
    const rootY = frames.map(f => (f.joints.left_hip.y + f.joints.right_hip.y) / 2);
    const smoothRootX = smoothSignal(rootX, 5);
    const smoothRootY = smoothSignal(rootY, 5);

    // Virtual Mass for Kinetic Energy (UE4/5 Physical Animation standard weights approx)
    const ARM_MASS = 1.5; // kg (Virtual infant weight)
    const LEG_MASS = 2.0; 

    for (let i = 2; i < frames.length; i++) {
      const dt = 0.1; // Assumed fixed time step 

      // -- CARTESIAN KINEMATICS --
      const p2 = { x: smoothX[i], y: smoothY[i], z: 0 };
      const p1 = { x: smoothX[i-1], y: smoothY[i-1], z: 0 };
      const vel = UEMath.dist(p1, p2) / dt;
      
      const p0 = { x: smoothX[i-2], y: smoothY[i-2], z: 0 };
      const prevVel = UEMath.dist(p0, p1) / dt;
      const accel = (vel - prevVel) / dt;
      
      let jerk = 0;
      if (i > 2) {
          const pm1 = { x: smoothX[i-3], y: smoothY[i-3], z: 0 };
          const prevPrevVel = UEMath.dist(pm1, p0) / dt;
          const prevAccel = (prevVel - prevPrevVel) / dt;
          jerk = (accel - prevAccel) / dt;
      }
      
      // -- UE5 PHYSICS: ROOT MOTION FLUX --
      // Calculate Root velocity vector
      const r2 = { x: smoothRootX[i], y: smoothRootY[i], z: 0 };
      const r1 = { x: smoothRootX[i-1], y: smoothRootY[i-1], z: 0 };
      const rootVel = UEMath.dist(r1, r2) / dt;
      
      // Flux = Deviation from center stability. High flux = Core instability or high activity.
      // We look at the derivative of root velocity (Root Acceleration/Force)
      const rootStress = Math.abs(rootVel);

      // -- UE5 PHYSICS: KINETIC ENERGY (Rigid Body Dynamics) --
      // Ek = 0.5 * m * v^2
      // We sum energies of extremities to get "Global Activation Energy"
      // Right Wrist
      const ek_rw = 0.5 * ARM_MASS * Math.pow(vel, 2);
      // Left Wrist
      const lw2 = { x: frames[i].joints.left_wrist.x, y: frames[i].joints.left_wrist.y, z:0 };
      const lw1 = { x: frames[i-1].joints.left_wrist.x, y: frames[i-1].joints.left_wrist.y, z:0 };
      const vel_lw = UEMath.dist(lw1, lw2) / dt;
      const ek_lw = 0.5 * ARM_MASS * Math.pow(vel_lw, 2);
      // Legs
      const rk2 = { x: frames[i].joints.right_ankle.x, y: frames[i].joints.right_ankle.y, z:0 };
      const rk1 = { x: frames[i-1].joints.right_ankle.x, y: frames[i-1].joints.right_ankle.y, z:0 };
      const vel_rk = UEMath.dist(rk1, rk2) / dt;
      const ek_rk = 0.5 * LEG_MASS * Math.pow(vel_rk, 2);
      
      const total_ke = ek_rw + ek_lw + ek_rk;

      // -- UE5 KINEMATICS: ANGULAR JERK --
      // Calculate Angle of Elbow Chain: Shoulder -> Elbow -> Wrist
      // We track the rate of change of this angle (Angular Velocity) -> then Accel -> then Jerk
      const angleCurrent = UEMath.angle(
          UEMath.vector(frames[i].joints.right_elbow, frames[i].joints.right_shoulder),
          UEMath.vector(frames[i].joints.right_elbow, frames[i].joints.right_wrist)
      );
      const anglePrev = UEMath.angle(
          UEMath.vector(frames[i-1].joints.right_elbow, frames[i-1].joints.right_shoulder),
          UEMath.vector(frames[i-1].joints.right_elbow, frames[i-1].joints.right_wrist)
      );
      const anglePrev2 = UEMath.angle(
          UEMath.vector(frames[i-2].joints.right_elbow, frames[i-2].joints.right_shoulder),
          UEMath.vector(frames[i-2].joints.right_elbow, frames[i-2].joints.right_wrist)
      );

      const angVel = (angleCurrent - anglePrev) / dt;
      const angVelPrev = (anglePrev - anglePrev2) / dt;
      const angAccel = (angVel - angVelPrev) / dt;
      // We treat angAccel variation as "Angular Smoothness/Jerk" proxy
      // Lower is smoother. High = clonus/tremor.
      const angularSmoothness = Math.abs(angAccel);

      // Dimensionless Jerk (Log scale for UI)
      const absJerk = Math.abs(jerk);
      const logJerk = absJerk > 0 ? Math.log(absJerk) : -5;
      const fluencyMetric = Math.max(0, Math.min(10, jerkOffset + logJerk));

      const windowStart = Math.max(0, i - windowSize);
      const windowDataRaw = rawX.slice(windowStart, i + 1);

      // Recurrence Rate
      const recurrenceRate = calculateRecurrenceRate(windowDataRaw, entropyR * 10); 
      // HFD
      const hfd = calculateFractalDimension(windowDataRaw, 5);
      
      metrics.push({
        timestamp: frames[i].timestamp,
        entropy: calculateSampleEntropy(windowDataRaw, 2, entropyR),
        fluency_velocity: vel,
        fluency_jerk: fluencyMetric,
        fractal_dim: hfd,
        phase_x: p2.x,
        phase_v: vel * (p2.x > p1.x ? 1 : -1),
        // New UE Metrics
        kinetic_energy: total_ke,
        angular_jerk: angularSmoothness,
        root_stress: rootStress
      });
    }
    return metrics;
  },

  detectCrampedSynchronized: (frames: SkeletonFrame[], fps: number = 10): { riskScore: number, jointDetails: any } => {
    const signals: Record<string, number[]> = {
        'right_knee': [], 'left_knee': [],
        'right_elbow': [], 'left_elbow': []
    };

    frames.forEach(frame => {
        signals['right_knee'].push(calculateCosineSimilarity(frame.joints.right_knee, frame.joints.right_hip, frame.joints.right_ankle));
        signals['left_knee'].push(calculateCosineSimilarity(frame.joints.left_knee, frame.joints.left_hip, frame.joints.left_ankle));
        signals['right_elbow'].push(calculateCosineSimilarity(frame.joints.right_elbow, frame.joints.right_shoulder, frame.joints.right_wrist));
        signals['left_elbow'].push(calculateCosineSimilarity(frame.joints.left_elbow, frame.joints.left_shoulder, frame.joints.left_wrist));
    });

    const lagsSeconds = [0.5, 1.0]; 
    let riskCount = 0;
    const details: any = {};
    const lowerLimbs = ['right_knee', 'left_knee'];
    const THRESHOLD = 0.6; 

    for (const [joint, signal] of Object.entries(signals)) {
        const corrValues = [];
        for (const sec of lagsSeconds) {
            const lagFrames = Math.floor(sec * fps);
            const s1 = signal.slice(0, signal.length - lagFrames);
            const s2 = signal.slice(lagFrames);
            corrValues.push(calculatePearsonCorrelation(s1, s2));
        }
        
        const avgCorr = corrValues.reduce((a,b) => a+b, 0) / corrValues.length;
        details[joint] = avgCorr;

        if (avgCorr > THRESHOLD) {
            if (lowerLimbs.includes(joint)) riskCount += 1; 
        }
    }

    return {
        riskScore: Math.min(1, riskCount / 2),
        jointDetails: details
    };
  },

  calculatePostureMetrics: (frames: SkeletonFrame[], movementData?: MovementMetrics[]): PostureMetrics => {
    if (frames.length === 0) {
        return { shoulder_flexion_index: 0, hip_flexion_index: 0, symmetry_score: 1, tone_label: 'Normal', frog_leg_score: 0, spontaneous_activity: 0, sustained_posture_score: 0, crying_index: 0, eye_openness_index: 0, arousal_index: 0, state_transition_probability: 0 };
    }

    let totalVel = 0;
    let stillFrames = 0;
    const ACTIVITY_THRESH = 1.0; 
    const frameVelocities: number[] = [];

    let totalShoulderFlex = 0;
    let totalHipFlex = 0;

    for(let i=1; i<frames.length; i++) {
        const dL = UEMath.dist(frames[i].joints.left_wrist, frames[i-1].joints.left_wrist);
        const dR = UEMath.dist(frames[i].joints.right_wrist, frames[i-1].joints.right_wrist);
        const dLK = UEMath.dist(frames[i].joints.left_ankle, frames[i-1].joints.left_ankle);
        const dRK = UEMath.dist(frames[i].joints.right_ankle, frames[i-1].joints.right_ankle);
        
        const frameVel = (dL + dR + dLK + dRK);
        frameVelocities.push(frameVel);
        totalVel += frameVel;

        if (frameVel < ACTIVITY_THRESH) {
            stillFrames++;
        }

        const sL = UEMath.angle(UEMath.vector(frames[i].joints.left_shoulder, frames[i].joints.left_elbow), UEMath.vector(frames[i].joints.left_shoulder, frames[i].joints.left_hip));
        const sR = UEMath.angle(UEMath.vector(frames[i].joints.right_shoulder, frames[i].joints.right_elbow), UEMath.vector(frames[i].joints.right_shoulder, frames[i].joints.right_hip));
        
        const hL = UEMath.angle(UEMath.vector(frames[i].joints.left_hip, frames[i].joints.left_knee), UEMath.vector(frames[i].joints.left_hip, frames[i].joints.left_shoulder));
        const hR = UEMath.angle(UEMath.vector(frames[i].joints.right_hip, frames[i].joints.right_knee), UEMath.vector(frames[i].joints.right_hip, frames[i].joints.right_shoulder));

        totalShoulderFlex += (Math.max(0, 180 - sL)/135 + Math.max(0, 180 - sR)/135)/2;
        totalHipFlex += (Math.max(0, 180 - hL)/135 + Math.max(0, 180 - hR)/135)/2;
    }
    
    const spontaneous_activity = (totalVel / frames.length) * 10;
    const sustained_posture_score = stillFrames / (frames.length - 1);
    
    const activeStates = frameVelocities.map(v => v > ACTIVITY_THRESH * 2 ? 1 : 0);
    let transitionCount = 0;
    let quietStateCount = 0;
    let arousalAccumulator = 0;
    let totalTransitions = 0;

    for(let i=1; i<activeStates.length; i++) {
        if (activeStates[i-1] === 0) {
            quietStateCount++;
            if (activeStates[i] === 1) {
                transitionCount++;
                if (movementData && movementData[i]) {
                    arousalAccumulator += movementData[i].entropy;
                } else {
                    arousalAccumulator += frameVelocities[i]; 
                }
                totalTransitions++;
            }
        }
    }

    const state_transition_probability = quietStateCount > 0 ? transitionCount / quietStateCount : 0;
    const arousal_index = totalTransitions > 0 ? (movementData ? arousalAccumulator/totalTransitions : (arousalAccumulator/totalTransitions)/50) : 0;

    let mouthVar = 0;
    let mouthVals = [];
    for(let i=0; i<frames.length; i++) {
        const nose = frames[i].joints.nose;
        const l_mouth = frames[i].joints.left_mouth;
        const r_mouth = frames[i].joints.right_mouth;
        
        if (nose && l_mouth && r_mouth) {
            const midMouthZ = (l_mouth.y + r_mouth.y)/2;
            const dist = midMouthZ - nose.y; 
            mouthVals.push(dist);
        }
    }
    if (mouthVals.length > 5) {
        const meanMouth = mouthVals.reduce((a,b)=>a+b,0) / mouthVals.length;
        const varMouth = mouthVals.reduce((a,b)=>a+Math.pow(b-meanMouth,2),0) / mouthVals.length;
        mouthVar = Math.sqrt(varMouth);
    }
    const crying_index = Math.min(1, mouthVar * 2.0);

    let eyeVar = 0;
    let eyeVals = [];
    for(let i=0; i<frames.length; i++) {
        const nose = frames[i].joints.nose;
        const l_eye = frames[i].joints.left_eye;
        const r_eye = frames[i].joints.right_eye;
        
        if (nose && l_eye && r_eye) {
            const eyeCenterX = (l_eye.x + r_eye.x)/2;
            const dist = eyeCenterX - nose.x;
            eyeVals.push(dist);
        }
    }
    if (eyeVals.length > 5) {
        const meanEye = eyeVals.reduce((a,b)=>a+b,0) / eyeVals.length;
        const varEye = eyeVals.reduce((a,b)=>a+Math.pow(b-meanEye,2),0) / eyeVals.length;
        eyeVar = Math.sqrt(varEye);
    }
    const eye_openness_index = Math.min(1, eyeVar * 1.5); 

    let frogSum = 0;
    for (let i = 0; i < frames.length; i++) {
        const l_knee = frames[i].joints.left_knee;
        const r_knee = frames[i].joints.right_knee;
        const l_hip = frames[i].joints.left_hip;
        const r_hip = frames[i].joints.right_hip;
        
        if (l_knee && r_knee && l_hip && r_hip) {
            const hipDist = Math.abs(l_hip.x - r_hip.x);
            const kneeDist = Math.abs(l_knee.x - r_knee.x);
            const splayRatio = hipDist > 0 ? kneeDist / hipDist : 1;
            const splay = Math.min(1, Math.max(0, (splayRatio - 1.2) / 2));
            frogSum += splay;
        }
    }
    const frog_leg_score = frogSum / frames.length;

    const avgFlex = (totalShoulderFlex/frames.length + totalHipFlex/frames.length) / 2;
    let tone_label: 'Normal' | 'Hypotonic' | 'Hypertonic' = 'Normal';
    if (avgFlex < 0.4) tone_label = 'Hypotonic';
    if (avgFlex > 0.8) tone_label = 'Hypertonic';

    return {
        shoulder_flexion_index: totalShoulderFlex / frames.length,
        hip_flexion_index: totalHipFlex / frames.length,
        symmetry_score: 1, 
        tone_label,
        frog_leg_score,
        spontaneous_activity,
        sustained_posture_score,
        crying_index,
        eye_openness_index,
        arousal_index,
        state_transition_probability
    };
  },

  detectSeizureSignatures: (frames: SkeletonFrame[], movementData: MovementMetrics[], config?: MotionConfig): SeizureMetrics => {
     if (frames.length < 30) return { rhythmicity_score: 0, stiffness_score: 0, eye_deviation_score: 0, dominant_frequency: 0, limb_synchrony: 0, calculated_type: 'None' };

     const rhythmWeight = config?.rhythmicityWeight ?? 0.7;
     const stiffnessDivisor = config?.stiffnessThreshold ?? 0.6;

     const velocities = movementData.map(m => m.fluency_velocity);
     const { dominantFreq, powerRatio } = calculateFrequencyAnalysis(velocities);
     const peakReg = calculatePeakRegularity(velocities);

     const eyeOffsets: number[] = [];
     frames.forEach(f => {
         if (f.joints.nose && f.joints.left_eye && f.joints.right_eye) {
            const midEye = (f.joints.left_eye.x + f.joints.right_eye.x) / 2;
            eyeOffsets.push(midEye - f.joints.nose.x);
         }
     });
     const meanOffset = eyeOffsets.reduce((a,b)=>a+b,0)/eyeOffsets.length;
     const varOffset = eyeOffsets.reduce((a,b)=>a+Math.pow(b-meanOffset,2),0)/eyeOffsets.length;
     
     const deviationMag = Math.abs(meanOffset);
     const stability = 1 / (1 + varOffset);
     const eye_deviation_score = Math.min(1, (deviationMag > 2.0 ? 1 : 0) * stability);

     const stiffness_score = Math.min(1, 1 / (1 + (movementData.reduce((a,b)=>a+b.fluency_velocity,0)/movementData.length) / stiffnessDivisor));

     const rightArmVel = frames.map((f, i) => i>0 ? UEMath.dist(f.joints.right_wrist, frames[i-1].joints.right_wrist) : 0).slice(1);
     const leftArmVel = frames.map((f, i) => i>0 ? UEMath.dist(f.joints.left_wrist, frames[i-1].joints.left_wrist) : 0).slice(1);
     const limb_synchrony = calculatePearsonCorrelation(rightArmVel, leftArmVel);

     const posture = physicsEngine.calculatePostureMetrics(frames, movementData); 
     const isCrying = posture.crying_index > 0.4;

     let type: 'None' | 'Clonic' | 'Tonic' | 'Myoclonic' = 'None';
     let rhythmicity_score = 0;
     const isBreathing = dominantFreq < 1.5;

     if (isCrying) {
         type = 'None';
         rhythmicity_score = 0; 
     } else if (isBreathing) {
         type = 'None';
         rhythmicity_score = 0; 
     } else {
         const inSeizureBand = dominantFreq >= 1.5 && dominantFreq <= 5.0;
         const isTremor = dominantFreq > 5.0; 
         
         rhythmicity_score = (powerRatio * 0.4 + peakReg * 0.6) * rhythmWeight;
         
         const highJerkCount = movementData.filter(m => m.fluency_jerk > 8.5).length;
         const isMyoclonicCandidate = highJerkCount > 0 && highJerkCount < (frames.length * 0.1); 

         if (stiffness_score > 0.8 || eye_deviation_score > 0.7) {
             type = 'Tonic';
         }
         else if (inSeizureBand && rhythmicity_score > 0.5) {
             type = 'Clonic';
         }
         else if (isMyoclonicCandidate && !isTremor && !inSeizureBand) {
             type = 'Myoclonic';
         }
     }

     return {
         rhythmicity_score,
         stiffness_score,
         eye_deviation_score,
         dominant_frequency: dominantFreq,
         limb_synchrony,
         calculated_type: type
     };
  }
};
