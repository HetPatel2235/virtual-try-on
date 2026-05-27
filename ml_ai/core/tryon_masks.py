"""
Build inpaint masks for IDM-VTON when the public API cannot auto-mask lower body.

The Hugging Face space always uses upper_body when is_checked=True.
For pants/jeans we paint a leg mask onto the person image and pass it as ImageEditor layers.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IDM_WIDTH = 768
IDM_HEIGHT = 1024


def _kp_xy(keypoints, name: str, min_conf: float = 0.25):
    for kp in keypoints:
        if kp.name == name and kp.confidence >= min_conf:
            return int(kp.x), int(kp.y)
    return None


def legs_detected(keypoints) -> bool:
    """True if hips and at least one ankle are visible (full-body photo)."""
    has_hips = _kp_xy(keypoints, "left_hip") and _kp_xy(keypoints, "right_hip")
    has_ankle = _kp_xy(keypoints, "left_ankle") or _kp_xy(keypoints, "right_ankle")
    return bool(has_hips and has_ankle)


def _leg_polygon(
    hip, knee, ankle, inner_hip, height: int, width: int, expand: int = 28
) -> np.ndarray | None:
    pts = [p for p in (hip, knee, ankle, inner_hip) if p is not None]
    if len(pts) < 3:
        return None
    poly = np.array(pts, dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if expand > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
        mask = cv2.dilate(mask, k, iterations=3)
    return mask


def _fallback_lower_mask(width: int, height: int) -> np.ndarray:
    """Use lower ~62% of frame when legs are not detected."""
    mask = np.zeros((height, width), dtype=np.uint8)
    y0 = int(height * 0.34)
    x0 = int(width * 0.10)
    x1 = int(width * 0.90)
    mask[y0:, x0:x1] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    return cv2.dilate(mask, k, iterations=2)


def build_lower_body_mask_gray(
    person_bgr: np.ndarray,
    width: int = IDM_WIDTH,
    height: int = IDM_HEIGHT,
) -> tuple[np.ndarray, bool]:
    """
    Single-channel mask (255 = inpaint legs/pants region).
    Returns (mask, legs_visible).
    """
    from ml_ai.core.mediapipe_real import create_real_pose_detector

    scaled = cv2.resize(person_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    detector = create_real_pose_detector()
    try:
        pose = detector.detect_pose(scaled)
    finally:
        detector.release()

    visible = legs_detected(pose.keypoints)
    mask = np.zeros((height, width), dtype=np.uint8)
    kps = pose.keypoints

    lh = _kp_xy(kps, "left_hip")
    rh = _kp_xy(kps, "right_hip")
    lk = _kp_xy(kps, "left_knee")
    rk = _kp_xy(kps, "right_knee")
    la = _kp_xy(kps, "left_ankle")
    ra = _kp_xy(kps, "right_ankle")

    if lh and rh:
        cx = (lh[0] + rh[0]) // 2
        cy = (lh[1] + rh[1]) // 2
        inner = (cx, cy)
    else:
        inner = None

    left_poly = _leg_polygon(lh, lk, la, inner or rh, height, width)
    right_poly = _leg_polygon(rh, rk, ra, inner or lh, height, width)

    if left_poly is not None:
        mask = np.maximum(mask, left_poly)
    if right_poly is not None:
        mask = np.maximum(mask, right_poly)

    if lh and rh:
        top_y = min(lh[1], rh[1]) - int(height * 0.04)
        bot_y = min(lh[1], rh[1]) + int(height * 0.12)
        left_x = min(lh[0], rh[0]) - int(width * 0.12)
        right_x = max(lh[0], rh[0]) + int(width * 0.12)
        cv2.rectangle(mask, (left_x, top_y), (right_x, bot_y), 255, -1)

    if mask.max() == 0:
        logger.warning("Could not detect legs; using fallback lower-body mask.")
        mask = _fallback_lower_mask(width, height)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.dilate(mask, k, iterations=2)
    return mask, visible


def build_image_editor_layer(
    person_bgr: np.ndarray,
    width: int = IDM_WIDTH,
    height: int = IDM_HEIGHT,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Build ImageEditor layer: person photo with white mask painted on leg region.
    IDM-VTON reads white pixels from this layer when is_checked=False.

    Returns (layer_bgr, mask_gray, legs_visible).
    """
    scaled = cv2.resize(person_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    mask, visible = build_lower_body_mask_gray(person_bgr, width, height)
    layer = scaled.copy()
    layer[mask > 127] = (255, 255, 255)
    return layer, mask, visible


def build_lower_body_mask_image(
    person_bgr: np.ndarray,
    width: int = IDM_WIDTH,
    height: int = IDM_HEIGHT,
) -> np.ndarray:
    """White-on-black BGR mask (legacy/debug)."""
    mask, _ = build_lower_body_mask_gray(person_bgr, width, height)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
