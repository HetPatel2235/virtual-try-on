"""Create placeholder lower-body garments (jeans, cargos, chinos) for the catalog."""

import json
from pathlib import Path

import cv2
import numpy as np

GARMENTS = [
    {
        "id": "jeans-001",
        "name": "Classic Slim Jeans",
        "category": "jeans",
        "brand": "DenimCo",
        "description": "Mid-rise slim fit denim jeans",
        "material": "98% Cotton, 2% Elastane",
        "price_usd": 49.99,
        "available_colors": ["indigo", "black", "light wash"],
        "fill_bgr": (45, 65, 120),
    },
    {
        "id": "cargo-001",
        "name": "Utility Cargo Pants",
        "category": "cargo pants",
        "brand": "FieldWear",
        "description": "Relaxed fit cargo pants with side pockets",
        "material": "100% Cotton Twill",
        "price_usd": 59.99,
        "available_colors": ["olive", "khaki", "black"],
        "fill_bgr": (50, 90, 55),
    },
    {
        "id": "chinos-001",
        "name": "Smart Chinos",
        "category": "chinos",
        "brand": "FormalEdge",
        "description": "Tapered fit chino trousers",
        "material": "97% Cotton, 3% Spandex",
        "price_usd": 44.99,
        "available_colors": ["navy", "stone", "charcoal"],
        "fill_bgr": (75, 95, 130),
    },
]

SIZE_CHART = {
    "28": {"waist_circumference_cm": 71.0, "hip_circumference_cm": 88.0, "inseam_length_cm": 76.0},
    "30": {"waist_circumference_cm": 76.0, "hip_circumference_cm": 93.0, "inseam_length_cm": 78.0},
    "32": {"waist_circumference_cm": 81.0, "hip_circumference_cm": 98.0, "inseam_length_cm": 80.0},
    "34": {"waist_circumference_cm": 86.0, "hip_circumference_cm": 103.0, "inseam_length_cm": 81.0},
    "36": {"waist_circumference_cm": 91.0, "hip_circumference_cm": 108.0, "inseam_length_cm": 82.0},
}


def draw_pants_image(fill_bgr: tuple[int, int, int]) -> np.ndarray:
    """Simple flat-lay pants silhouette on transparent canvas (768x1024)."""
    h, w = 1024, 768
    img = np.zeros((h, w, 4), dtype=np.uint8)

    waist_y = int(h * 0.12)
    crotch_y = int(h * 0.38)
    hem_y = int(h * 0.92)
    cx = w // 2
    waist_half = int(w * 0.22)
    knee_half = int(w * 0.19)
    hem_half = int(w * 0.17)
    gap = int(w * 0.04)

    left_pts = np.array([
        [cx - waist_half, waist_y],
        [cx - gap // 2, waist_y],
        [cx - gap // 2, crotch_y],
        [cx - knee_half, crotch_y + 80],
        [cx - hem_half, hem_y],
        [cx - waist_half + 20, hem_y],
        [cx - waist_half, crotch_y],
    ], dtype=np.int32)
    right_pts = np.array([
        [cx + waist_half, waist_y],
        [cx + gap // 2, waist_y],
        [cx + gap // 2, crotch_y],
        [cx + knee_half, crotch_y + 80],
        [cx + hem_half, hem_y],
        [cx + waist_half - 20, hem_y],
        [cx + waist_half, crotch_y],
    ], dtype=np.int32)

    b, g, r = fill_bgr
    color = (b, g, r, 255)
    cv2.fillPoly(img, [left_pts, right_pts], color)
    cv2.rectangle(img, (cx - waist_half, waist_y - 8), (cx + waist_half, waist_y + 24), color, -1)
    return img


def main():
    base = Path("database/data/garments")
    for g in GARMENTS:
        out_dir = base / g["id"]
        out_dir.mkdir(parents=True, exist_ok=True)

        pants = draw_pants_image(g["fill_bgr"])
        cv2.imwrite(str(out_dir / "image.png"), pants)

        meta = {k: v for k, v in g.items() if k != "fill_bgr"}
        meta["image_filename"] = "image.png"
        meta["size_chart"] = SIZE_CHART
        meta["anchor_points"] = {
            "waist_left": [0.28, 0.12],
            "waist_right": [0.72, 0.12],
            "crotch": [0.50, 0.38],
            "left_knee": [0.30, 0.62],
            "right_knee": [0.70, 0.62],
            "left_hem": [0.28, 0.92],
            "right_hem": [0.72, 0.92],
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Created {g['id']}")


if __name__ == "__main__":
    main()
