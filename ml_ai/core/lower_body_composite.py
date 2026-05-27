"""
Local lower-body try-on: warp garment onto legs using pose keypoints.
Works offline and always produces a visible change (no Hugging Face required).
"""

from __future__ import annotations

import cv2
import numpy as np

from ml_ai.core.image_io import prepare_image_for_api
from ml_ai.core.tryon_masks import build_lower_body_mask_gray


def _kp(keypoints, name: str, w: int, h: int, default_frac: tuple[float, float]):
    for kp in keypoints:
        if kp.name == name and kp.confidence >= 0.25:
            return int(kp.x), int(kp.y)
    return int(default_frac[0] * w), int(default_frac[1] * h)


def composite_lower_body(
    person_bgr: np.ndarray,
    garment_bgr: np.ndarray,
    keypoints,
    blend_alpha: float = 0.92,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warp garment into the lower-body region and blend onto the person.

    Returns (composite_bgr, warped_garment_bgr, garment_mask_uint8).
    """
    person = prepare_image_for_api(person_bgr)
    garment = prepare_image_for_api(garment_bgr)
    h, w = person.shape[:2]

    lh = _kp(keypoints, "left_hip", w, h, (0.38, 0.42))
    rh = _kp(keypoints, "right_hip", w, h, (0.62, 0.42))
    la = _kp(keypoints, "left_ankle", w, h, (0.38, 0.92))
    ra = _kp(keypoints, "right_ankle", w, h, (0.62, 0.92))

    y_top = min(lh[1], rh[1]) - int(h * 0.02)
    y_bot = max(la[1], ra[1]) + int(h * 0.02)
    y_top = max(0, y_top)
    y_bot = min(h - 1, y_bot)
    if y_bot - y_top < int(h * 0.2):
        y_top = int(h * 0.34)
        y_bot = int(h * 0.96)

    cx = (lh[0] + rh[0]) // 2
    half_w = int(max(abs(rh[0] - lh[0]) * 1.15, w * 0.22))
    x0 = max(0, cx - half_w)
    x1 = min(w - 1, cx + half_w)

    # Slight trapezoid (wider at hips)
    hip_half = int((x1 - x0) * 0.52)
    ankle_half = int((x1 - x0) * 0.42)
    dst = np.float32([
        [cx - hip_half, y_top],
        [cx + hip_half, y_top],
        [cx + ankle_half, y_bot],
        [cx - ankle_half, y_bot],
    ])

    gh, gw = garment.shape[:2]
    src = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(garment, matrix, (w, h), flags=cv2.INTER_LINEAR)

    mask_region, _ = build_lower_body_mask_gray(person, width=w, height=h)
    warped_mask = cv2.warpPerspective(
        np.ones((gh, gw), dtype=np.uint8) * 255,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
    )
    blend_mask = ((warped_mask > 40) & (mask_region > 127)).astype(np.uint8) * 255
    blend_mask = cv2.GaussianBlur(blend_mask, (9, 9), 0)
    m = (blend_mask.astype(np.float32) / 255.0 * blend_alpha)[:, :, np.newaxis]

    composite = (person.astype(np.float32) * (1.0 - m) + warped.astype(np.float32) * m)
    return composite.astype(np.uint8), warped, blend_mask
