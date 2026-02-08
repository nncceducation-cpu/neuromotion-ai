
import numpy as np
from typing import List, Dict, Any

# --- PHYSICS MATH HELPER ---

def distance(p1: dict, p2: dict) -> float:
    return np.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2 + (p2['z'] - p1['z'])**2)

def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot_prod = np.dot(v1, v2)
    cosine = np.clip(dot_prod / (norm1 * norm2), -1.0, 1.0)
    return np.arccos(cosine) * (180.0 / np.pi)

def smooth_signal(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    if len(data) < window_size:
        return data
    kernel = np.ones(window_size) / window_size
    # Mode 'valid' returns window-adjusted size, we pad to keep same length or handle edge
    # Simplified to valid for analysis
    return np.convolve(data, kernel, mode='same')

def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    if len(data) < 10: return 0.0
    
    # Normalize
    std = np.std(data)
    mean = np.mean(data)
    if std == 0: return 0.0
    normalized = (data - mean) / std
    
    N = len(normalized)
    threshold = r
    
    def get_count(length):
        count = 0
        for i in range(N - length):
            template = normalized[i:i+length]
            # Vectorized search
            # (In production, use KDTree or optimized C implementation)
            # Brute force python is O(N^2), acceptable for <3000 frames
            for j in range(N - length):
                if i == j: continue
                segment = normalized[j:j+length]
                dist = np.max(np.abs(template - segment))
                if dist < threshold:
                    count += 1
        return count

    A = get_count(m + 1)
    B = get_count(m)
    
    if B == 0: return 0.0
    if A == 0: return -np.log(1/B)
    return -np.log(A/B)

def fractal_dimension(data: np.ndarray, kmax: int = 5) -> float:
    if len(data) < 10: return 1.5
    N = len(data)
    Lk = []
    
    for k in range(1, kmax + 1):
        Lm = 0
        for m in range(k):
            Lmk = 0
            n_max = int((N - m - 1) / k)
            for i in range(1, n_max + 1):
                Lmk += np.abs(data[m + i * k] - data[m + (i - 1) * k])
            norm = (N - 1) / (n_max * k)
            Lmk = Lmk * norm / k
            Lm += Lmk
        Lk.append(Lm / k)
    
    # Linear Regression of ln(Lk) vs ln(1/k)
    x = np.log(1.0 / np.array(range(1, kmax + 1)))
    y = np.log(np.array(Lk))
    
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope # HFD

# --- MAIN ENGINE ---

def process_frames(frames: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not frames: return []
    
    # Extract Coordinates
    timestamps = [f.get('timestamp', i*0.1) for i, f in enumerate(frames)]
    right_wrist_x = np.array([f['joints']['right_wrist']['x'] for f in frames])
    right_wrist_y = np.array([f['joints']['right_wrist']['y'] for f in frames])
    
    # Root CoM (Hips)
    root_x = np.array([(f['joints']['left_hip']['x'] + f['joints']['right_hip']['x']) / 2 for f in frames])
    root_y = np.array([(f['joints']['left_hip']['y'] + f['joints']['right_hip']['y']) / 2 for f in frames])
    
    # Smoothing
    smooth_x = smooth_signal(right_wrist_x)
    smooth_y = smooth_signal(right_wrist_y)
    smooth_root_x = smooth_signal(root_x)
    smooth_root_y = smooth_signal(root_y)
    
    dt = 0.1 # Fixed step
    metrics = []
    
    # Config Params
    window_size = config.get('windowSize', 30)
    entropy_r = config.get('entropyThreshold', 0.2)
    jerk_threshold = config.get('jerkThreshold', 5.0)
    
    ARM_MASS = 1.5
    
    for i in range(2, len(frames)):
        # Kinematics
        dx = smooth_x[i] - smooth_x[i-1]
        dy = smooth_y[i] - smooth_y[i-1]
        dist = np.sqrt(dx**2 + dy**2)
        vel = dist / dt
        
        # Accel
        prev_dist = np.sqrt((smooth_x[i-1] - smooth_x[i-2])**2 + (smooth_y[i-1] - smooth_y[i-2])**2)
        prev_vel = prev_dist / dt
        accel = (vel - prev_vel) / dt
        
        # Jerk
        jerk = 0
        if i > 2:
            prev_prev_dist = np.sqrt((smooth_x[i-2] - smooth_x[i-3])**2 + (smooth_y[i-2] - smooth_y[i-3])**2)
            prev_prev_vel = prev_prev_dist / dt
            prev_accel = (prev_vel - prev_prev_vel) / dt
            jerk = (accel - prev_accel) / dt
            
        # Root Stress
        dr_x = smooth_root_x[i] - smooth_root_x[i-1]
        dr_y = smooth_root_y[i] - smooth_root_y[i-1]
        root_vel = np.sqrt(dr_x**2 + dr_y**2) / dt
        root_stress = np.abs(root_vel)
        
        # Kinetic Energy (Simplified to just Right Wrist for demo parity)
        ek = 0.5 * ARM_MASS * (vel**2)
        
        # Log Jerk Metric
        abs_jerk = np.abs(jerk)
        log_jerk = np.log(abs_jerk) if abs_jerk > 0 else -5
        fluency = max(0, min(10, jerk_threshold + log_jerk))
        
        # Entropy & Fractal Windows
        window_start = max(0, i - window_size)
        window_data = right_wrist_x[window_start:i+1]
        
        samp_en = sample_entropy(window_data, 2, entropy_r)
        hfd = fractal_dimension(window_data, 5)
        
        metrics.append({
            'timestamp': timestamps[i],
            'entropy': float(samp_en),
            'fluency_velocity': float(vel),
            'fluency_jerk': float(fluency),
            'fractal_dim': float(hfd),
            'phase_x': float(smooth_x[i]),
            'phase_v': float(vel * (1 if dx > 0 else -1)),
            'kinetic_energy': float(ek),
            'angular_jerk': 0.0, # Simplified
            'root_stress': float(root_stress)
        })
        
    return metrics
