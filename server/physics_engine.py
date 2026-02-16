
import numpy as np
from typing import List, Dict, Any, Tuple

# --- PHYSICS MATH HELPERS ---

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
    return np.convolve(data, kernel, mode='same')

def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    if len(data) < 10: return 0.0
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
    x = np.log(1.0 / np.array(range(1, kmax + 1)))
    y = np.log(np.array(Lk))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope

def joint_angle(proximal: dict, mid: dict, distal: dict) -> float:
    v1 = np.array([proximal['x'] - mid['x'], proximal['y'] - mid['y']])
    v2 = np.array([distal['x'] - mid['x'], distal['y'] - mid['y']])
    return vector_angle(v1, v2)

def extract_xy(frames: List[Dict], joint_name: str):
    x = np.array([f['joints'][joint_name]['x'] for f in frames])
    y = np.array([f['joints'][joint_name]['y'] for f in frames])
    return x, y

def frame_velocity(sx: np.ndarray, sy: np.ndarray, i: int, dt: float) -> float:
    dx = sx[i] - sx[i-1]
    dy = sy[i] - sy[i-1]
    return np.sqrt(dx**2 + dy**2) / dt


# Joint names used for visibility extraction
_TRACKED_JOINTS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

def extract_frame_visibility(frame: Dict[str, Any]) -> Tuple[float, float]:
    """Extract average and minimum visibility across tracked joints for a single frame."""
    visibilities = []
    joints = frame.get('joints', {})
    for name in _TRACKED_JOINTS:
        joint = joints.get(name, {})
        visibilities.append(joint.get('visibility', 1.0))
    if not visibilities:
        return 1.0, 1.0
    return float(np.mean(visibilities)), float(np.min(visibilities))


# Segment mass fractions (approximate neonatal proportions, per side)
SEGMENT_MASS = {
    'shoulder': 0.15,
    'hip': 0.25,
    'knee': 0.10,
    'ankle': 0.05,
    'wrist': 0.05,
}
# Total: 2 sides * (0.15+0.25+0.10+0.05+0.05) = 1.20 — we normalize below
_MASS_TOTAL = 2 * sum(SEGMENT_MASS.values())

# Segment mass for kinetic energy (kg, approximate neonatal)
ARM_MASS = 0.3    # per arm (forearm+hand)
LEG_MASS = 0.5    # per leg (shank+foot)


# --- POSTURE & SEIZURE METRIC COMPUTATION ---

def compute_posture_metrics(
    frames: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute posture assessment metrics from skeleton frames and per-frame metrics."""
    if not metrics or not frames:
        return _default_posture()

    # Shoulder flexion: angle at shoulder (neck_midpoint -> shoulder -> elbow)
    shoulder_flex_vals = []
    hip_flex_vals = []
    frog_leg_vals = []
    for f in frames:
        j = f['joints']
        # Shoulder midpoint as proxy for neck
        neck_mid = {
            'x': (j['left_shoulder']['x'] + j['right_shoulder']['x']) / 2.0,
            'y': (j['left_shoulder']['y'] + j['right_shoulder']['y']) / 2.0,
        }
        # Bilateral shoulder flexion
        sf_r = joint_angle(neck_mid, j['right_shoulder'], j['right_elbow'])
        sf_l = joint_angle(neck_mid, j['left_shoulder'], j['left_elbow'])
        shoulder_flex_vals.append((sf_r + sf_l) / 2.0)

        # Hip flexion: angle at hip (shoulder -> hip -> knee)
        hf_r = joint_angle(j['right_shoulder'], j['right_hip'], j['right_knee'])
        hf_l = joint_angle(j['left_shoulder'], j['left_hip'], j['left_knee'])
        hip_flex_vals.append((hf_r + hf_l) / 2.0)

        # Frog leg: hip abduction (angle between legs via hip midpoint)
        hip_mid = {
            'x': (j['left_hip']['x'] + j['right_hip']['x']) / 2.0,
            'y': (j['left_hip']['y'] + j['right_hip']['y']) / 2.0,
        }
        frog = joint_angle(j['left_knee'], hip_mid, j['right_knee'])
        frog_leg_vals.append(frog)

    avg_shoulder_flex = np.mean(shoulder_flex_vals) / 180.0  # normalize to 0-1
    avg_hip_flex = np.mean(hip_flex_vals) / 180.0
    # Frog leg score: lower angle = more splayed = higher score
    avg_frog_angle = np.mean(frog_leg_vals)
    frog_leg_score = max(0.0, 1.0 - (avg_frog_angle / 180.0))

    # Symmetry from per-frame bilateral_symmetry
    symmetries = [m.get('bilateral_symmetry', 0.5) for m in metrics]
    symmetry_score = float(np.mean(symmetries))

    # Tone label from entropy + jerk heuristic
    entropies = [m.get('entropy', 0.4) for m in metrics]
    jerks = [m.get('fluency_jerk', 5.0) for m in metrics]
    avg_entropy = float(np.mean(entropies))
    avg_jerk = float(np.mean(jerks))

    if avg_entropy < 0.2 and avg_jerk < 3.0:
        tone_label = "Hypotonic"
    elif avg_jerk > 8.0:
        tone_label = "Hypertonic"
    else:
        tone_label = "Normal"

    # Spontaneous activity = mean kinetic energy
    kes = [m.get('kinetic_energy', 0) for m in metrics]
    spontaneous_activity = float(np.mean(kes))

    # Sustained posture score: high = static/lethargic
    ke_std = float(np.std(kes))
    ke_mean = float(np.mean(kes))
    sustained_posture_score = 1.0 - (ke_std / (ke_mean + 1e-6)) if ke_mean > 0 else 1.0
    sustained_posture_score = max(0.0, min(1.0, sustained_posture_score))

    # Arousal index: magnitude of entropy spike above mean
    arousal_index = float(np.max(entropies) - avg_entropy) if entropies else 0.0

    # State transition probability: fraction of frames where KE crosses mean
    transitions = 0
    for idx in range(1, len(kes)):
        if (kes[idx] >= ke_mean) != (kes[idx - 1] >= ke_mean):
            transitions += 1
    state_transition = transitions / max(1, len(kes) - 1)

    return {
        "shoulder_flexion_index": float(np.clip(avg_shoulder_flex, 0, 1)),
        "hip_flexion_index": float(np.clip(avg_hip_flex, 0, 1)),
        "symmetry_score": symmetry_score,
        "tone_label": tone_label,
        "frog_leg_score": float(np.clip(frog_leg_score, 0, 1)),
        "spontaneous_activity": spontaneous_activity,
        "sustained_posture_score": sustained_posture_score,
        "crying_index": 0.0,  # Cannot compute — YOLO has no mouth keypoints
        "eye_openness_index": 0.0,  # Cannot compute — YOLO has no eyelid landmarks
        "arousal_index": arousal_index,
        "state_transition_probability": float(state_transition),
    }


def compute_seizure_metrics(
    frames: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute seizure-specific biomarkers from skeleton frames and per-frame metrics."""
    if not metrics or len(metrics) < 10 or not frames:
        return _default_seizure()

    dt = 0.1  # 10 FPS
    rhythmicity_weight = config.get('rhythmicityWeight', 0.7)
    stiffness_threshold = config.get('stiffnessThreshold', 0.6)
    sensitivity = config.get('sensitivity', 0.85)

    # --- FFT of right wrist Y for rhythmicity and dominant frequency ---
    rw_y = np.array([f['joints']['right_wrist']['y'] for f in frames])
    rw_y_detrended = rw_y - np.mean(rw_y)
    n = len(rw_y_detrended)
    # Zero-pad for better frequency resolution
    n_fft = max(256, 2 ** int(np.ceil(np.log2(n))))
    fft_vals = np.abs(np.fft.rfft(rw_y_detrended, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=dt)

    # Total power (exclude DC component)
    total_power = np.sum(fft_vals[1:] ** 2)
    if total_power < 1e-10:
        return _default_seizure()

    # Seizure band power (1.5-5 Hz)
    seizure_mask = (freqs >= 1.5) & (freqs <= 5.0)
    seizure_power = np.sum(fft_vals[seizure_mask] ** 2)
    rhythmicity_raw = seizure_power / total_power
    rhythmicity_score = float(np.clip(rhythmicity_raw * rhythmicity_weight, 0, 1))

    # Dominant frequency (peak in spectrum excluding DC)
    peak_idx = np.argmax(fft_vals[1:]) + 1
    dominant_frequency = float(freqs[peak_idx])

    # --- Stiffness score from entropy variance ---
    entropies = np.array([m.get('entropy', 0.4) for m in metrics])
    entropy_var = float(np.var(entropies))
    stiffness_score = float(np.clip(
        1.0 / (1.0 + entropy_var / max(stiffness_threshold, 0.01)), 0, 1
    ))

    # --- Eye deviation score ---
    eye_dev = 0.0
    nose_present = any(
        f.get('joints', {}).get('nose') is not None and
        f.get('joints', {}).get('left_eye') is not None
        for f in frames
    )
    if nose_present:
        deviations = []
        for f in frames:
            j = f['joints']
            nose = j.get('nose')
            left_eye = j.get('left_eye')
            if nose and left_eye:
                deviations.append(abs(left_eye['x'] - nose['x']))
        if deviations:
            eye_dev = float(np.mean(deviations) / 10.0)  # normalize to ~0-1 range
            eye_dev = min(1.0, eye_dev)

    # --- Limb synchrony: cross-correlation of L/R wrist Y-velocities ---
    lw_y = np.array([f['joints']['left_wrist']['y'] for f in frames])
    rw_vel_y = np.diff(rw_y) / dt
    lw_vel_y = np.diff(lw_y) / dt
    if len(rw_vel_y) > 2:
        # Normalize to avoid scale dependency
        rw_std = np.std(rw_vel_y)
        lw_std = np.std(lw_vel_y)
        if rw_std > 1e-6 and lw_std > 1e-6:
            corr = np.corrcoef(rw_vel_y, lw_vel_y)[0, 1]
            limb_synchrony = float(np.clip((corr + 1.0) / 2.0, 0, 1))  # map [-1,1] to [0,1]
        else:
            limb_synchrony = 0.5
    else:
        limb_synchrony = 0.5

    # --- Seizure type classification (rule-based) ---
    jerks = [m.get('fluency_jerk', 5.0) for m in metrics]
    avg_jerk = float(np.mean(jerks))
    peak_jerk = float(np.max(jerks)) if jerks else 0.0
    mean_jerk = float(np.mean(jerks)) if jerks else 0.0

    if 1.5 <= dominant_frequency <= 5.0 and rhythmicity_score > 0.5 * rhythmicity_weight:
        calculated_type = "Clonic"
    elif stiffness_score > 0.7 and rhythmicity_score < 0.3:
        calculated_type = "Tonic"
    elif peak_jerk > 3.0 * mean_jerk and mean_jerk > 0 and rhythmicity_score < 0.3:
        calculated_type = "Myoclonic"
    else:
        calculated_type = "None"

    return {
        "rhythmicity_score": rhythmicity_score,
        "stiffness_score": stiffness_score,
        "eye_deviation_score": eye_dev,
        "dominant_frequency": dominant_frequency,
        "limb_synchrony": limb_synchrony,
        "calculated_type": calculated_type,
    }


def _default_posture() -> Dict[str, Any]:
    return {
        "shoulder_flexion_index": 0.0, "hip_flexion_index": 0.0,
        "symmetry_score": 1.0, "tone_label": "Normal",
        "frog_leg_score": 0.0, "spontaneous_activity": 0.0,
        "sustained_posture_score": 0.0,
        "crying_index": 0.0, "eye_openness_index": 0.0,
        "arousal_index": 0.0, "state_transition_probability": 0.0,
    }


def _default_seizure() -> Dict[str, Any]:
    return {
        "rhythmicity_score": 0.0, "stiffness_score": 0.0,
        "eye_deviation_score": 0.0, "dominant_frequency": 0.0,
        "limb_synchrony": 0.0, "calculated_type": "None",
    }


# --- MAIN ENGINE ---

def process_frames(
    frames: List[Dict[str, Any]], config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Process skeleton frames into per-frame movement metrics + posture + seizure summaries.

    Returns:
        (metrics_list, posture_metrics, seizure_metrics)
    """
    if not frames:
        return [], _default_posture(), _default_seizure()

    timestamps = [f.get('timestamp', i * 0.1) for i, f in enumerate(frames)]

    # Extract all joint coordinates
    rw_x, rw_y = extract_xy(frames, 'right_wrist')
    lw_x, lw_y = extract_xy(frames, 'left_wrist')
    rs_x, rs_y = extract_xy(frames, 'right_shoulder')
    ls_x, ls_y = extract_xy(frames, 'left_shoulder')
    rh_x, rh_y = extract_xy(frames, 'right_hip')
    lh_x, lh_y = extract_xy(frames, 'left_hip')
    rk_x, rk_y = extract_xy(frames, 'right_knee')
    lk_x, lk_y = extract_xy(frames, 'left_knee')
    ra_x, ra_y = extract_xy(frames, 'right_ankle')
    la_x, la_y = extract_xy(frames, 'left_ankle')

    # Nose (optional — may be missing in some frames)
    nose_x = np.array([
        f['joints']['nose']['x'] if f.get('joints', {}).get('nose') is not None else 0.0
        for f in frames
    ])
    nose_y = np.array([
        f['joints']['nose']['y'] if f.get('joints', {}).get('nose') is not None else 0.0
        for f in frames
    ])
    has_nose = np.any(nose_x != 0) or np.any(nose_y != 0)

    # Bilateral wrist average (for entropy/fractal)
    bilateral_wrist_x = (rw_x + lw_x) / 2.0

    # Root CoM (hip midpoint)
    root_x = (lh_x + rh_x) / 2.0
    root_y = (lh_y + rh_y) / 2.0

    # Shoulder midpoint (for head stability reference)
    shoulder_mid_x = (ls_x + rs_x) / 2.0
    shoulder_mid_y = (ls_y + rs_y) / 2.0

    # Weighted whole-body CoM
    joints_for_com = [
        (rs_x, rs_y, 'shoulder'), (ls_x, ls_y, 'shoulder'),
        (rh_x, rh_y, 'hip'),     (lh_x, lh_y, 'hip'),
        (rk_x, rk_y, 'knee'),    (lk_x, lk_y, 'knee'),
        (ra_x, ra_y, 'ankle'),   (la_x, la_y, 'ankle'),
        (rw_x, rw_y, 'wrist'),   (lw_x, lw_y, 'wrist'),
    ]
    com_x = np.zeros(len(frames))
    com_y = np.zeros(len(frames))
    for jx, jy, seg in joints_for_com:
        w = SEGMENT_MASS[seg] / _MASS_TOTAL
        com_x += jx * w
        com_y += jy * w

    # Smooth all signals
    s_rw_x, s_rw_y = smooth_signal(rw_x), smooth_signal(rw_y)
    s_lw_x, s_lw_y = smooth_signal(lw_x), smooth_signal(lw_y)
    s_ra_x, s_ra_y = smooth_signal(ra_x), smooth_signal(ra_y)
    s_la_x, s_la_y = smooth_signal(la_x), smooth_signal(la_y)
    s_root_x, s_root_y = smooth_signal(root_x), smooth_signal(root_y)
    s_com_x, s_com_y = smooth_signal(com_x), smooth_signal(com_y)
    s_nose_x, s_nose_y = smooth_signal(nose_x), smooth_signal(nose_y)
    s_shoulder_mid_x = smooth_signal(shoulder_mid_x)
    s_shoulder_mid_y = smooth_signal(shoulder_mid_y)

    # Compute joint angle time series (for angular jerk & ROM)
    elbow_angles_r = np.array([
        joint_angle(f['joints']['right_shoulder'], f['joints']['right_elbow'], f['joints']['right_wrist'])
        for f in frames
    ])
    elbow_angles_l = np.array([
        joint_angle(f['joints']['left_shoulder'], f['joints']['left_elbow'], f['joints']['left_wrist'])
        for f in frames
    ])
    knee_angles_r = np.array([
        joint_angle(f['joints']['right_hip'], f['joints']['right_knee'], f['joints']['right_ankle'])
        for f in frames
    ])
    knee_angles_l = np.array([
        joint_angle(f['joints']['left_hip'], f['joints']['left_knee'], f['joints']['left_ankle'])
        for f in frames
    ])
    s_elbow_r = smooth_signal(elbow_angles_r)
    s_elbow_l = smooth_signal(elbow_angles_l)

    dt = 0.1
    metrics = []

    window_size = config.get('windowSize', 30)
    entropy_r = config.get('entropyThreshold', 0.2)
    jerk_threshold = config.get('jerkThreshold', 5.0)

    EPS = 1e-6

    for i in range(2, len(frames)):
        # --- Per-frame visibility/confidence ---
        avg_vis, min_vis = extract_frame_visibility(frames[i])

        # --- Right wrist kinematics ---
        rw_vel = frame_velocity(s_rw_x, s_rw_y, i, dt)
        rw_prev_vel = frame_velocity(s_rw_x, s_rw_y, i-1, dt) if i >= 2 else 0

        # --- Left wrist kinematics ---
        lw_vel = frame_velocity(s_lw_x, s_lw_y, i, dt)

        # Bilateral average velocity (for fluency)
        avg_wrist_vel = (rw_vel + lw_vel) / 2.0

        # --- Right wrist acceleration & jerk ---
        rw_accel = (rw_vel - rw_prev_vel) / dt
        rw_jerk = 0.0
        if i > 2:
            rw_prev_prev_vel = frame_velocity(s_rw_x, s_rw_y, i-2, dt)
            rw_prev_accel = (rw_prev_vel - rw_prev_prev_vel) / dt
            rw_jerk = (rw_accel - rw_prev_accel) / dt

        # --- Left wrist acceleration & jerk ---
        lw_prev_vel = frame_velocity(s_lw_x, s_lw_y, i-1, dt) if i >= 2 else 0
        lw_accel = (lw_vel - lw_prev_vel) / dt
        lw_jerk = 0.0
        if i > 2:
            lw_prev_prev_vel = frame_velocity(s_lw_x, s_lw_y, i-2, dt)
            lw_prev_accel = (lw_prev_vel - lw_prev_prev_vel) / dt
            lw_jerk = (lw_accel - lw_prev_accel) / dt

        avg_jerk = (rw_jerk + lw_jerk) / 2.0

        # --- Bilateral Symmetry (wrists) ---
        symmetry_denom = rw_vel + lw_vel + EPS
        bilateral_symmetry = 1.0 - abs(rw_vel - lw_vel) / symmetry_denom

        # --- Ankle kinematics (lower limb) ---
        ra_vel = frame_velocity(s_ra_x, s_ra_y, i, dt)
        la_vel = frame_velocity(s_la_x, s_la_y, i, dt)
        lower_limb_ke = 0.5 * LEG_MASS * (ra_vel**2 + la_vel**2)

        # --- Kinetic energy (both wrists + both ankles) ---
        upper_ke = 0.5 * ARM_MASS * (rw_vel**2 + lw_vel**2)
        total_ke = upper_ke + lower_limb_ke

        # --- Root Stress (hip midpoint velocity) ---
        root_vel = frame_velocity(s_root_x, s_root_y, i, dt)
        root_stress = abs(root_vel)

        # --- Whole-body CoM velocity ---
        com_vel = frame_velocity(s_com_x, s_com_y, i, dt)

        # --- Head Stability (nose velocity relative to shoulder midpoint) ---
        if has_nose:
            rel_nose_dx = (s_nose_x[i] - s_shoulder_mid_x[i]) - (s_nose_x[i-1] - s_shoulder_mid_x[i-1])
            rel_nose_dy = (s_nose_y[i] - s_shoulder_mid_y[i]) - (s_nose_y[i-1] - s_shoulder_mid_y[i-1])
            head_stab = np.sqrt(rel_nose_dx**2 + rel_nose_dy**2) / dt
        else:
            head_stab = 0.0

        # --- Angular Jerk (from elbow angle time series) ---
        ang_vel = (s_elbow_r[i] - s_elbow_r[i-1]) / dt
        ang_vel_l = (s_elbow_l[i] - s_elbow_l[i-1]) / dt
        angular_jerk_val = 0.0
        if i > 2:
            prev_ang_vel = (s_elbow_r[i-1] - s_elbow_r[i-2]) / dt
            ang_accel = (ang_vel - prev_ang_vel) / dt
            prev_ang_vel_l = (s_elbow_l[i-1] - s_elbow_l[i-2]) / dt
            ang_accel_l = (ang_vel_l - prev_ang_vel_l) / dt
            if i > 3:
                prev_prev_ang_vel = (s_elbow_r[i-2] - s_elbow_r[i-3]) / dt
                prev_ang_accel = (prev_ang_vel - prev_prev_ang_vel) / dt
                ang_jerk_r = (ang_accel - prev_ang_accel) / dt

                prev_prev_ang_vel_l = (s_elbow_l[i-2] - s_elbow_l[i-3]) / dt
                prev_ang_accel_l = (prev_ang_vel_l - prev_prev_ang_vel_l) / dt
                ang_jerk_l = (ang_accel_l - prev_ang_accel_l) / dt

                angular_jerk_val = (abs(ang_jerk_r) + abs(ang_jerk_l)) / 2.0

        # --- Fluency (log jerk metric, bilateral average) ---
        abs_jerk = abs(avg_jerk)
        log_jerk = np.log(abs_jerk) if abs_jerk > 0 else -5
        fluency = max(0, min(10, jerk_threshold + log_jerk))

        # --- Entropy & Fractal (bilateral wrist signal) ---
        window_start = max(0, i - window_size)
        window_data = bilateral_wrist_x[window_start:i+1]
        samp_en = sample_entropy(window_data, 2, entropy_r)
        hfd = fractal_dimension(window_data, 5)

        # --- Phase Space (right wrist trajectory for single-joint phase portrait) ---
        rw_dx = s_rw_x[i] - s_rw_x[i-1]

        metrics.append({
            'timestamp': timestamps[i],
            'entropy': float(samp_en),
            'fluency_velocity': float(avg_wrist_vel),
            'fluency_jerk': float(fluency),
            'fractal_dim': float(hfd),
            'phase_x': float(s_rw_x[i]),
            'phase_v': float(rw_vel * (1 if rw_dx > 0 else -1)),
            'kinetic_energy': float(total_ke),
            'angular_jerk': float(angular_jerk_val),
            'root_stress': float(root_stress),
            'bilateral_symmetry': float(bilateral_symmetry),
            'lower_limb_kinetic_energy': float(lower_limb_ke),
            'com_velocity': float(com_vel),
            'head_stability': float(head_stab),
            'avg_visibility': avg_vis,
            'min_visibility': min_vis,
        })

    # Compute ROM summaries (appended to last frame for aggregation convenience)
    if metrics:
        metrics[-1]['_elbow_rom'] = float(
            (np.max(elbow_angles_r) - np.min(elbow_angles_r)) +
            (np.max(elbow_angles_l) - np.min(elbow_angles_l))
        ) / 2.0
        metrics[-1]['_knee_rom'] = float(
            (np.max(knee_angles_r) - np.min(knee_angles_r)) +
            (np.max(knee_angles_l) - np.min(knee_angles_l))
        ) / 2.0

    # Compute summary metrics
    posture = compute_posture_metrics(frames, metrics, config)
    seizure = compute_seizure_metrics(frames, metrics, config)

    return metrics, posture, seizure
