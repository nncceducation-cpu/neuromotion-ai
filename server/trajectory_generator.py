"""
Deterministic trajectory generation and clinical profiles.
Converted from constants.ts — full kinematics solver for infant motion simulation.

Produces the same output as the TypeScript version for identical seeds.
"""

import math
import random
from typing import List, Dict, Any, Optional, Callable


# --- DETERMINISTIC RANDOM (PRNG) ---
# Same algorithm as TypeScript version for reproducibility

_seed = 123456


def set_seed(s: int):
    global _seed
    _seed = abs(s) % 233280


def _seeded_random() -> float:
    global _seed
    _seed = (_seed * 9301 + 49297) % 233280
    return _seed / 233280


# --- RAW GEOMETRY GENERATORS (FULL BODY) ---

def _get_chaotic_val(t: float) -> float:
    return math.sin(t) + 0.5 * math.sin(2.23 * t)


def _get_rhythmic_val(t: float) -> float:
    return math.sin(3 * t)


def _get_tremor_val(t: float) -> float:
    return (_seeded_random() - 0.5) * 0.8 + math.sin(t * 15) * 0.2


def _get_seizure_val(t: float) -> float:
    return math.sin(t * 15) * 0.8


# Movement types
MovementType = str  # 'CHAOTIC' | 'RHYTHMIC' | 'TREMOR' | 'FLACCID' | 'SEIZURE'


def _make_point(x: float, y: float, z: float, stability: str) -> Dict[str, float]:
    """Create a Point3D dict with visibility based on stability."""
    v_base = 0.98
    if stability == "MED":
        v_base = 0.90
    elif stability == "LOW":
        v_base = 0.85

    v = min(1.0, max(0.4, v_base + (_seeded_random() * 0.1 - 0.05)))
    return {"x": x, "y": y, "z": z, "visibility": v}


def _solve_limb(
    root: Dict[str, float],
    len1: float,
    len2: float,
    phase: float,
    is_leg: bool,
    side: str,
    t: float,
    move_fn: Callable[[float], float],
    movement_type: str,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Kinematics solver for a single limb chain (2 segments).
    Returns (mid_joint, end_joint) as Point3D dicts.
    """
    side_mult = 1 if side == "L" else -1

    val1 = move_fn(t + phase)
    if movement_type in ("RHYTHMIC", "FLACCID", "SEIZURE"):
        val2 = val1
    else:
        val2 = move_fn(t + phase + 1.5)

    # Modifiers
    if movement_type == "TREMOR":
        val1 = val1 * 0.5
        val2 = val2 * 2.0
    if movement_type == "SEIZURE":
        val1 = val1 * 0.8
        val2 = val2 * 0.8

    # 1. Proximal Joint Rotation (Shoulder/Hip)
    theta = 0.0
    phi = 0.0

    if is_leg:
        if movement_type == "FLACCID":
            theta = 0.8
            phi = 0.8
        elif movement_type == "SEIZURE":
            theta = 0.5 + (abs(val1) * 0.2)
            phi = 0.3
        else:
            theta = 1.6 + (val1 * 0.4)
            phi = 0.4 + (val2 * 0.15)
    else:
        if movement_type == "FLACCID":
            theta = 0.5
            phi = 0.2
        elif movement_type == "SEIZURE":
            theta = 2.0 + (abs(val1) * 0.2)
            phi = 0.5
        else:
            theta = 2.4 + (val1 * 0.35)
            phi = 0.7 + (val2 * 0.25)

    mid_x = root["x"] + (math.sin(phi) * len1 * side_mult)
    mid_y = root["y"] + (math.cos(theta) * len1)
    mid_z = abs(math.sin(val1)) * 15
    mid = _make_point(mid_x, mid_y, mid_z, "MED")

    # 2. Distal Joint Rotation
    flex = 0.0
    if is_leg:
        if movement_type == "SEIZURE":
            flex = 0.3 + (val2 * 0.1)
        elif movement_type == "FLACCID":
            flex = 0.2
        else:
            flex = 1.5 + (val2 * 0.3)
    else:
        if movement_type == "SEIZURE":
            flex = -2.0 + (val2 * 0.1)
        elif movement_type == "FLACCID":
            flex = 0.2
        else:
            flex = -1.8 + (val2 * 0.3)

    angle2 = theta + flex
    end_x = mid["x"] + (math.sin(phi * 0.8) * len1 * side_mult)
    end_y = mid["y"] + (math.cos(angle2) * len2)
    end_z = abs(math.sin(val2)) * 10
    end = _make_point(end_x, end_y, end_z, "LOW")

    return mid, end


def build_frame(
    t: float,
    move_fn: Callable[[float], float],
    movement_type: str,
) -> Dict[str, Any]:
    """
    Build a complete SkeletonFrame from a time value and movement function.
    Implements infant anthropometry with anatomically realistic proportions.
    """
    # Global Body Movement (Breathing/Writhing)
    breath = math.sin(t * 0.2) * 0.1 if movement_type == "FLACCID" else math.sin(t * 0.5) * 0.5
    writhing = 0.0 if movement_type == "FLACCID" else math.cos(t * 0.3) * 1.5

    # Center point
    cx = 50 + (0 if movement_type == "SEIZURE" else writhing)
    neck_y = 30 + breath

    # Proportions
    torso_len = 22
    hip_y = neck_y + torso_len
    shoulder_width = 14
    pelvis_width = 12
    upper_arm = 9
    forearm = 8
    thigh = 11
    shin = 9

    # HEAD & EYES
    rot = 0 if movement_type == "FLACCID" else (0.02 if movement_type == "SEIZURE" else math.sin(t * 0.2) * 0.05)
    nose = _make_point(cx, 15 + breath, -20, "HIGH")

    # Eye Deviation Logic
    eye_offset_x = 0.0
    if movement_type == "SEIZURE":
        eye_offset_x = 4.0
    elif movement_type in ("CHAOTIC", "RHYTHMIC"):
        eye_offset_x = math.sin(t * 0.8) * 2

    left_eye = _make_point(cx - 3 + eye_offset_x, 12 + breath, -15, "MED")
    right_eye = _make_point(cx + 3 + eye_offset_x, 12 + breath, -15, "MED")

    # MOUTH (Crying Detection)
    mouth_open = 0.0
    if movement_type in ("CHAOTIC", "TREMOR"):
        mouth_open = abs(math.sin(t * 1.5)) * 1.5
    elif movement_type == "FLACCID":
        mouth_open = 0.1
    else:
        mouth_open = 0.2

    left_mouth = _make_point(cx - 1.5, 18 + breath + mouth_open, -18, "MED")
    right_mouth = _make_point(cx + 1.5, 18 + breath + mouth_open, -18, "MED")

    # Shoulders
    l_sh = _make_point(cx + shoulder_width / 2, neck_y - (rot * 5), 0, "HIGH")
    r_sh = _make_point(cx - shoulder_width / 2, neck_y + (rot * 5), 0, "HIGH")

    # Hips
    l_hip = _make_point(cx + pelvis_width / 2, hip_y + (rot * 5), 0, "HIGH")
    r_hip = _make_point(cx - pelvis_width / 2, hip_y - (rot * 5), 0, "HIGH")

    # Limbs
    la_mid, la_end = _solve_limb(l_sh, upper_arm, forearm, 0, False, "L", t, move_fn, movement_type)

    ra_phase = 0 if movement_type in ("RHYTHMIC", "SEIZURE") else 2.1
    ra_mid, ra_end = _solve_limb(r_sh, upper_arm, forearm, ra_phase, False, "R", t, move_fn, movement_type)

    ll_phase = 0 if movement_type in ("RHYTHMIC", "SEIZURE") else 1.3
    ll_mid, ll_end = _solve_limb(l_hip, thigh, shin, ll_phase, True, "L", t, move_fn, movement_type)

    rl_phase = 0 if movement_type in ("RHYTHMIC", "SEIZURE") else 3.5
    rl_mid, rl_end = _solve_limb(r_hip, thigh, shin, rl_phase, True, "R", t, move_fn, movement_type)

    return {
        "timestamp": t,
        "joints": {
            "nose": nose,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "left_mouth": left_mouth,
            "right_mouth": right_mouth,
            "left_shoulder": l_sh,
            "right_shoulder": r_sh,
            "left_hip": l_hip,
            "right_hip": r_hip,
            "left_elbow": la_mid,
            "left_wrist": la_end,
            "right_elbow": ra_mid,
            "right_wrist": ra_end,
            "left_knee": ll_mid,
            "left_ankle": ll_end,
            "right_knee": rl_mid,
            "right_ankle": rl_end,
        },
    }


# --- Trajectory Generators ---

def generate_normal_continuous(frames: int) -> List[Dict[str, Any]]:
    return [build_frame(i * 0.1, _get_chaotic_val, "CHAOTIC") for i in range(frames)]


def generate_intermittent(frames: int) -> List[Dict[str, Any]]:
    data = []
    t = 0.0
    for i in range(frames):
        is_pause = math.sin(i * 0.05) > 0.5
        if not is_pause:
            t += 0.1
        data.append(build_frame(t, _get_chaotic_val, "CHAOTIC"))
    return data


def generate_hyperalert(frames: int) -> List[Dict[str, Any]]:
    return [build_frame(i * 0.1, _get_tremor_val, "TREMOR") for i in range(frames)]


def generate_flaccid(frames: int) -> List[Dict[str, Any]]:
    return [build_frame(i * 0.02, _get_chaotic_val, "FLACCID") for i in range(frames)]


def generate_seizure(frames: int) -> List[Dict[str, Any]]:
    return [build_frame(i * 0.1, _get_seizure_val, "SEIZURE") for i in range(frames)]


# --- Clinical Profiles ---

GENERATORS = {
    "NORMAL": generate_normal_continuous,
    "SARNAT_I": generate_hyperalert,
    "SARNAT_II": generate_intermittent,
    "SARNAT_III": generate_flaccid,
    "SEIZURE": generate_seizure,
}

CLINICAL_PROFILES = [
    {
        "id": "NORMAL",
        "label": "Normal",
        "features": {
            "entropyScore": "High",
            "fluencySAL": "Moderate",
            "fractalDimension": "High",
            "convexHullVolume": "Large",
            "phaseSpaceTopology": "Cloud",
            "clinicalNote": "Appropriate level of consciousness, normal tone (intermittent flexion), fluid movements. Crying and Spontaneous Eye Opening present.",
        },
    },
    {
        "id": "SARNAT_I",
        "label": "Sarnat Stage I",
        "features": {
            "entropyScore": "Very High",
            "fluencySAL": "Very Low (Tremors)",
            "fractalDimension": "High (Noise)",
            "convexHullVolume": "Restricted",
            "phaseSpaceTopology": "Dense Cloud",
            "clinicalNote": "Hyperalert, jittery, exaggerated reflexes, no seizures. Eyes wide open.",
        },
    },
    {
        "id": "SARNAT_II",
        "label": "Sarnat Stage II",
        "features": {
            "entropyScore": "Variable",
            "fluencySAL": "Low",
            "fractalDimension": "Moderate",
            "convexHullVolume": "Reduced",
            "phaseSpaceTopology": "Cluster",
            "clinicalNote": "Lethargic, hypotonic (extended), decreased activity with intermittent bursts.",
        },
    },
    {
        "id": "SARNAT_III",
        "label": "Sarnat Stage III",
        "features": {
            "entropyScore": "Near Zero",
            "fluencySAL": "Zero",
            "fractalDimension": "Flatline",
            "convexHullVolume": "Collapsed",
            "phaseSpaceTopology": "Point",
            "clinicalNote": "Stupor/Coma, flaccid tone, absent reflexes. No eye opening.",
        },
    },
    {
        "id": "SEIZURE",
        "label": "Seizures",
        "features": {
            "entropyScore": "Moderate/Periodic",
            "fluencySAL": "High Jerk + Stiffness",
            "fractalDimension": "Low",
            "convexHullVolume": "Moderate",
            "phaseSpaceTopology": "Limit Cycle",
            "clinicalNote": "Rhythmic clonic jerking with total body stiffness and sustained eye deviation.",
        },
    },
]


def get_profile_by_seed(seed: int) -> Dict[str, Any]:
    """Get a clinical profile deterministically from a seed value."""
    local_val = (seed * 9301 + 49297) % 233280
    normalized = local_val / 233280

    if normalized < 0.25:
        return CLINICAL_PROFILES[0]
    if normalized < 0.45:
        return CLINICAL_PROFILES[1]
    if normalized < 0.65:
        return CLINICAL_PROFILES[2]
    if normalized < 0.85:
        return CLINICAL_PROFILES[3]
    return CLINICAL_PROFILES[4]


def get_random_profile() -> Dict[str, Any]:
    """Get a random clinical profile."""
    rand = random.random()
    if rand < 0.25:
        return CLINICAL_PROFILES[0]
    if rand < 0.45:
        return CLINICAL_PROFILES[1]
    if rand < 0.65:
        return CLINICAL_PROFILES[2]
    if rand < 0.85:
        return CLINICAL_PROFILES[3]
    return CLINICAL_PROFILES[4]


def generate_trajectory(profile_id: str, frames: int = 300, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate a trajectory for a given clinical profile.
    If seed is provided, sets the PRNG seed for reproducibility.
    """
    if seed is not None:
        set_seed(seed)

    generator = GENERATORS.get(profile_id)
    if not generator:
        raise ValueError(f"Unknown profile_id: {profile_id}. Must be one of {list(GENERATORS.keys())}")

    return generator(frames)
