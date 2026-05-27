"""Image helpers for API / cloud pipelines."""

from __future__ import annotations

import cv2
import numpy as np


def prepare_image_for_api(image: np.ndarray) -> np.ndarray:
    """Convert BGRA or grayscale garment/person images to BGR for IDM-VTON."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        alpha = a.astype(np.float32) / 255.0
        rgb = np.stack([b, g, r], axis=-1).astype(np.float32)
        white = np.full_like(rgb, 255.0)
        return (rgb * alpha[..., None] + white * (1.0 - alpha[..., None])).astype(np.uint8)
    return image[:, :, :3].copy()
