"""
vitpose_inference.py - ViTPose + RTMDet inference module for Neuromotion-AI.
Handles: model loading, video frame extraction, person detection, pose estimation,
         and COCO keypoint -> SkeletonFrame conversion.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded model references (singleton)
_det_model = None
_pose_model = None
_models_loaded = False

# Paths
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
DEVICE = None  # Set at load time

# COCO keypoint index -> SkeletonFrame joint name
COCO_TO_SKELETON = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    # 3: left_ear  (not in SkeletonFrame)
    # 4: right_ear (not in SkeletonFrame)
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

REQUIRED_JOINTS = {
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
}


def _find_checkpoint(pattern: str) -> Optional[str]:
    """Find a checkpoint file in CHECKPOINT_DIR matching a pattern."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    for f in os.listdir(CHECKPOINT_DIR):
        if pattern in f and f.endswith(".pth"):
            return os.path.join(CHECKPOINT_DIR, f)
    return None


def _find_config(pattern: str) -> Optional[str]:
    """Find a config file in CHECKPOINT_DIR matching a pattern."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    for f in os.listdir(CHECKPOINT_DIR):
        if pattern in f and f.endswith(".py"):
            return os.path.join(CHECKPOINT_DIR, f)
    return None


def load_models() -> bool:
    """Load RTMDet detector and ViTPose pose estimator. Called once at startup."""
    global _det_model, _pose_model, _models_loaded, DEVICE

    if _models_loaded:
        return True

    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        from mmdet.apis import init_detector
        from mmpose.apis import init_model

        # Find configs and checkpoints
        det_config = _find_config("rtmdet")
        det_checkpoint = _find_checkpoint("rtmdet")
        pose_config = _find_config("ViTPose")
        pose_checkpoint = _find_checkpoint("ViTPose") or _find_checkpoint("vitpose")

        if not det_config or not det_checkpoint:
            logger.error(f"RTMDet config/checkpoint not found in {CHECKPOINT_DIR}")
            return False
        if not pose_config or not pose_checkpoint:
            logger.error(f"ViTPose config/checkpoint not found in {CHECKPOINT_DIR}")
            return False

        logger.info(f"Loading RTMDet from {det_config} on {DEVICE}...")
        _det_model = init_detector(det_config, det_checkpoint, device=DEVICE)

        logger.info(f"Loading ViTPose from {pose_config} on {DEVICE}...")
        _pose_model = init_model(pose_config, pose_checkpoint, device=DEVICE)

        _models_loaded = True
        logger.info("ViTPose models loaded successfully.")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"GPU memory allocated: {allocated:.2f} GB")

        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        _models_loaded = False
        return False


def is_loaded() -> bool:
    return _models_loaded


def coco_keypoints_to_skeleton_frame(
    keypoints: np.ndarray,
    keypoint_scores: np.ndarray,
    timestamp: float,
    img_width: int,
    img_height: int,
    min_visibility: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """Convert COCO 17-keypoint output to SkeletonFrame dict (coords normalized 0-100)."""
    joints = {}

    for coco_idx, joint_name in COCO_TO_SKELETON.items():
        x_pixel = float(keypoints[coco_idx][0])
        y_pixel = float(keypoints[coco_idx][1])
        score = float(keypoint_scores[coco_idx])

        joints[joint_name] = {
            "x": (x_pixel / img_width) * 100.0,
            "y": (y_pixel / img_height) * 100.0,
            "z": 0.0,
            "visibility": score,
        }

    # Reject frame if any required joint is below visibility threshold
    for rj in REQUIRED_JOINTS:
        if rj not in joints or joints[rj]["visibility"] < min_visibility:
            return None

    return {"timestamp": timestamp, "joints": joints}


def extract_frames_from_video(
    video_path: str, target_fps: float = 10.0, max_frames: int = 3000
) -> Tuple[List[np.ndarray], float, int, int]:
    """Extract frames from video at target FPS. Returns (frames, actual_fps, width, height)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    skip = max(1, int(round(video_fps / target_fps)))
    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    actual_fps = video_fps / skip
    logger.info(f"Extracted {len(frames)} frames (video FPS: {video_fps:.1f}, skip: {skip})")
    return frames, actual_fps, width, height


def detect_persons(frame: np.ndarray, score_threshold: float = 0.5) -> np.ndarray:
    """Run RTMDet person detection. Returns best bbox as (1, 4) xyxy or empty (0, 4)."""
    from mmdet.apis import inference_detector

    result = inference_detector(_det_model, frame)
    pred = result.pred_instances
    mask = (pred.labels == 0) & (pred.scores > score_threshold)
    bboxes = pred.bboxes[mask].cpu().numpy()
    scores = pred.scores[mask].cpu().numpy()

    if len(bboxes) == 0:
        return np.empty((0, 4))

    best = np.argmax(scores)
    return bboxes[best : best + 1]


def estimate_pose(
    frame: np.ndarray, bboxes: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run ViTPose on a frame. Returns (keypoints (17,2), scores (17,)) or None."""
    from mmpose.apis import inference_topdown

    if len(bboxes) == 0:
        return None

    results = inference_topdown(_pose_model, frame, bboxes=bboxes, bbox_format="xyxy")
    if not results:
        return None

    pred = results[0].pred_instances
    keypoints = pred.keypoints[0].cpu().numpy()
    keypoint_scores = pred.keypoint_scores[0].cpu().numpy()
    return keypoints, keypoint_scores


def process_video(
    video_path: str,
    target_fps: float = 10.0,
    det_score_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Full pipeline: video -> frames -> detection -> pose -> SkeletonFrames."""
    if not _models_loaded:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    frames, actual_fps, width, height = extract_frames_from_video(video_path, target_fps)
    if not frames:
        raise ValueError("No frames extracted from video")

    import torch

    skeleton_frames = []
    dt = 1.0 / actual_fps

    for i, frame_bgr in enumerate(frames):
        timestamp = i * dt

        bboxes = detect_persons(frame_bgr, det_score_threshold)
        if len(bboxes) == 0:
            continue

        pose_result = estimate_pose(frame_bgr, bboxes)
        if pose_result is None:
            continue

        keypoints, keypoint_scores = pose_result
        skeleton = coco_keypoints_to_skeleton_frame(
            keypoints, keypoint_scores, timestamp, width, height
        )
        if skeleton is not None:
            skeleton_frames.append(skeleton)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(frames)} frames")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Pipeline complete: {len(skeleton_frames)} valid frames from {len(frames)} extracted")
    return skeleton_frames
