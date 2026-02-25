"""
yolo_inference.py - YOLO26 Pose inference module for Neuromotion-AI.
Handles: model loading, video frame extraction, person detection + pose estimation,
         and COCO keypoint -> SkeletonFrame conversion.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Lazy-loaded model reference (singleton)
_model = None
_models_loaded = False
DEVICE = None

# COCO keypoint index -> SkeletonFrame joint name
COCO_TO_SKELETON = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
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


def load_models() -> bool:
    """Load YOLO26x-pose model. Called once at startup."""
    global _model, _models_loaded, DEVICE

    if _models_loaded:
        return True

    try:
        import torch
        from ultralytics import YOLO  # type: ignore[attr-defined]

        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading YOLO26x-pose on {DEVICE}...")
        _model = YOLO("yolo26x-pose.pt")
        _model.to(DEVICE)

        _models_loaded = True
        logger.info("YOLO26x-pose loaded successfully.")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"GPU memory allocated: {allocated:.2f} GB")

        return True
    except Exception as e:
        logger.error(f"Failed to load YOLO26x-pose: {e}", exc_info=True)
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


def process_video(
    video_path: str,
    target_fps: float = 10.0,
    det_score_threshold: float = 0.5,
    output_video_path: Optional[str] = None,
    frame_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Full pipeline: video -> frames -> YOLO26 detection+pose -> SkeletonFrames.

    If output_video_path is provided, writes an annotated MP4 with skeleton overlay.
    Returns (skeleton_frames, output_video_path or None).
    """
    if not _models_loaded:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    frames, actual_fps, width, height = extract_frames_from_video(video_path, target_fps)
    if not frames:
        raise ValueError("No frames extracted from video")

    import torch

    # Set up video writer for annotated output
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, actual_fps, (width, height))
        logger.info(f"Writing annotated video to {output_video_path}")

    skeleton_frames = []
    dt = 1.0 / actual_fps

    for i, frame_bgr in enumerate(frames):
        timestamp = i * dt

        # YOLO26 does detection + pose in a single forward pass
        results = _model(frame_bgr, verbose=False, conf=det_score_threshold)  # type: ignore[misc]
        result = results[0]

        # Write annotated frame (every frame, to keep video in sync)
        annotated = result.plot()
        if video_writer is not None:
            video_writer.write(annotated)

        # Send annotated frame to callback for live preview (every 3rd frame)
        if frame_callback is not None and i % 3 == 0:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_callback(annotated_rgb, i, len(frames))

        # Skip if no persons detected
        if result.keypoints is None or len(result.keypoints) == 0:
            continue

        # Pick the person with the largest bounding box area
        if result.boxes is not None and len(result.boxes) > 1:
            areas = (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]) * \
                    (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1])
            best_idx = int(areas.argmax())
        else:
            best_idx = 0

        # keypoints.data shape: (N, 17, 3) where 3 = (x_pixel, y_pixel, confidence)
        kpt_data = result.keypoints.data[best_idx].cpu().numpy()  # (17, 3)
        keypoints = kpt_data[:, :2]       # (17, 2) pixel coords
        keypoint_scores = kpt_data[:, 2]  # (17,) confidence

        skeleton = coco_keypoints_to_skeleton_frame(
            keypoints, keypoint_scores, timestamp, width, height
        )
        if skeleton is not None:
            skeleton_frames.append(skeleton)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(frames)} frames")

    if video_writer is not None:
        video_writer.release()
        logger.info(f"Annotated video saved: {output_video_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Pipeline complete: {len(skeleton_frames)} valid frames from {len(frames)} extracted")
    return skeleton_frames, output_video_path if video_writer is not None else None
