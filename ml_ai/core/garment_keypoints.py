"""
Garment Keypoints Module
AI-Based Virtual Try-On and Fit Recommendation System

Calibrated using real MediaPipe keypoint data:
    SW=278px, torso=404px (1.46xSW), image=1028x1370
    Left shoulder:(659,394), Right:(381,387)
    Left hip:(596,794), Right:(435,796)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GarmentAnchor:
    name: str
    garment_uv: Tuple[float, float]
    body_landmark: str | None = None
    offset_ratio: Tuple[float, float] = (0.0, 0.0)
    weight: int = 1


@dataclass
class GarmentKeypointSchema:
    category: str
    anchors: List[GarmentAnchor] = field(default_factory=list)

    def get_src_points(self, garment_w: int, garment_h: int, garment_mask: np.ndarray | None = None) -> np.ndarray:
        pts = []
        
        # If mask is provided, find the actual physical bounding box of the fabric
        bx, by, bw, bh = 0, 0, garment_w, garment_h
        if garment_mask is not None:
            import cv2
            coords = cv2.findNonZero(garment_mask)
            if coords is not None:
                bx, by, bw, bh = cv2.boundingRect(coords)

        for anchor in self.anchors:
            u, v = anchor.garment_uv
            # Map UV to the physical bounding box of the garment instead of the whole image
            x = bx + u * bw
            y = by + v * bh
            
            for _ in range(anchor.weight):
                pts.append([x, y])
        return np.array(pts, dtype=np.float32)

    def get_dst_points(
        self,
        keypoints: list,
        person_w: int,
        person_h: int,
        shoulder_scale: float = 1.0,
        person_mask: np.ndarray | None = None
    ) -> np.ndarray | None:
        """
        Derive destination control points dynamically on the person image.
        Uses MediaPipe keypoints + Torso segmentation mask (if available) for Perfect Fit.
        """
        kp_map = {kp.name: kp for kp in keypoints}
        derived = _derive_all_landmarks(kp_map, person_w, person_h, person_mask)
        if not derived:
            return None
        base_points = {**{k: (float(v.x), float(v.y)) for k, v in kp_map.items()}, **derived}

        # Apply shoulder_scale to shoulder width (cx centered)
        cx = person_w / 2
        for key in ["left_shoulder", "right_shoulder"]:
            if key in base_points:
                base_x, base_y = base_points[key]
                base_points[key] = (cx + (base_x - cx) * shoulder_scale, base_y)

        # ── PERFECT FIT LOGIC: Snap destination points to the exact torso contour ──
        if person_mask is not None:
            import cv2
            coords = cv2.findNonZero(person_mask)
            if coords is not None:
                bx, by, bw, bh = cv2.boundingRect(coords)
                
                # Determine which is viewer-left and viewer-right
                if "left_shoulder" in base_points and "right_shoulder" in base_points:
                    ls_x = base_points["left_shoulder"][0]
                    rs_x = base_points["right_shoulder"][0]
                    
                    if ls_x > rs_x:  # Front-facing (right shoulder is on viewer left)
                        base_points["right_shoulder_dst"] = (bx, base_points["right_shoulder"][1])
                        base_points["left_shoulder_dst"] = (bx + bw, base_points["left_shoulder"][1])
                    else:  # Back-facing
                        base_points["left_shoulder_dst"] = (bx, base_points["left_shoulder"][1])
                        base_points["right_shoulder_dst"] = (bx + bw, base_points["right_shoulder"][1])

                if "left_hip" in base_points and "right_hip" in base_points:
                    lh_x = base_points["left_hip"][0]
                    rh_x = base_points["right_hip"][0]
                    
                    if lh_x > rh_x:
                        base_points["right_hip_dst"] = (bx, base_points["right_hip"][1])
                        base_points["left_hip_dst"] = (bx + bw, base_points["left_hip"][1])
                    else:
                        base_points["left_hip_dst"] = (bx, base_points["left_hip"][1])
                        base_points["right_hip_dst"] = (bx + bw, base_points["right_hip"][1])

        pts = []
        for anchor in self.anchors:
            if not anchor.body_landmark:
                continue
            
            # Use dynamically overridden point if it exists, otherwise fall back to base skeleton
            lookup_name = anchor.body_landmark + "_dst"
            if lookup_name in base_points:
                px, py = base_points[lookup_name]
            elif anchor.body_landmark in base_points:
                px, py = base_points[anchor.body_landmark]
            else:
                continue

            # We no longer add garment UV offsets to the destination points.
            # The destination points are completely derived from the body proportions.
            for _ in range(anchor.weight):
                pts.append([px, py])

        if len(pts) != sum(a.weight for a in self.anchors if a.body_landmark):
            return None

        return np.array(pts, dtype=np.float32)

    def anchor_names(self) -> List[str]:
        return [a.name for a in self.anchors]


# ---------------------------------------------------------------------------
# Landmark derivation — calibrated to real body proportions
# Torso height ≈ 1.46 × SW (verified from keypoint data)
# ---------------------------------------------------------------------------

def _derive_all_landmarks(
    kp_map: dict,
    person_w: int,
    person_h: int,
    person_mask: np.ndarray | None = None
) -> Dict[str, Tuple[float, float]]:
    """
    Derive body landmarks using shoulder width (SW) as base unit.
    All proportions verified against real MediaPipe keypoint coordinates.
    """
    d: Dict[str, Tuple[float, float]] = {}

    def _get(name):
        kp = kp_map.get(name)
        return (float(kp.x), float(kp.y)) if kp else None

    ls   = _get("left_shoulder")
    rs   = _get("right_shoulder")
    lh   = _get("left_hip")
    rh   = _get("right_hip")
    le   = _get("left_elbow")
    re   = _get("right_elbow")
    lw   = _get("left_wrist")
    rw   = _get("right_wrist")
    lk   = _get("left_knee")
    rk   = _get("right_knee")
    la   = _get("left_ankle")
    ra   = _get("right_ankle")

    if not (ls and rs):
        return d

    # ── Guarantee viewer-relative coordinates ─────────────────────────
    if ls[0] > rs[0]:
        vl_s, vr_s = rs, ls     # Viewer-Left Shoulder, Viewer-Right Shoulder
        vl_h, vr_h = rh, lh     # Viewer-Left Hip, Viewer-Right Hip
        vl_e, vr_e = re, le     # Viewer-Left Elbow, Viewer-Right Elbow
        vl_w, vr_w = rw, lw     # Viewer-Left Wrist, Viewer-Right Wrist
        vl_k, vr_k = rk, lk
        vl_a, vr_a = ra, la
    else:
        vl_s, vr_s = ls, rs
        vl_h, vr_h = lh, rh
        vl_e, vr_e = le, re
        vl_w, vr_w = lw, rw
        vl_k, vr_k = lk, rk
        vl_a, vr_a = la, ra

    # ── Base metrics ──────────────────────────────────────────────────
    SY = (vl_s[1] + vr_s[1]) / 2.0
    SW = abs(vr_s[0] - vl_s[0])
    MX = vl_s[0] + SW / 2.0

    # ── Hips / Waist foundation ──────────────────────────────────────
    if vl_h and vr_h:
        hip_y  = (vl_h[1] + vr_h[1]) / 2.0
        hip_lx = vl_h[0]
        hip_rx = vr_h[0]
    elif vl_h:
        hip_y  = vl_h[1]
        hip_lx = vl_h[0]
        hip_rx = vr_s[0]
    elif vr_h:
        hip_y  = vr_h[1]
        hip_lx = vl_s[0]
        hip_rx = vr_h[0]
    else:
        hip_y  = SY + SW * 1.46
        hip_lx = vl_s[0]
        hip_rx = vr_s[0]

    # ── Neck / Collar (3-point curve) ────────────────────────────────
    # Base of the neck sits just slightly above the shoulder line
    neck_y = SY - SW * 0.08
    
    d["neck_center"]   = (MX, neck_y)
    d["collar_left"]   = (MX - SW * 0.15, neck_y)  # viewer-left
    d["collar_right"]  = (MX + SW * 0.15, neck_y)  # viewer-right
    d["collar_bottom"] = (MX, SY + SW * 0.05)      # dips below shoulder line

    def get_side_x(y_val):
        """
        Use the precise segmentation mask to get the exact left and right bounds 
        of the person's torso/shirt at a specific y-coordinate.
        If the mask isn't available, fall back to the heuristic approximation.
        """
        if person_mask is not None:
            import numpy as np
            y_int = int(round(y_val))
            y_int = max(0, min(person_mask.shape[0] - 1, y_int))
            row = person_mask[y_int, :]
            nonzero = np.nonzero(row)[0]
            if len(nonzero) > 0:
                return float(nonzero[0]), float(nonzero[-1])
        
        # Fallback to heuristic
        if hip_y <= SY:
            return vl_s[0] - SW * 0.10, vr_s[0] + SW * 0.10
        t = (y_val - SY) / (hip_y - SY)
        t = max(0.0, min(1.0, t))
        lx = vl_s[0] + t * (hip_lx - vl_s[0]) - SW * 0.12
        rx = vr_s[0] + t * (hip_rx - vr_s[0]) + SW * 0.12
        return lx, rx

    # ── Shoulder Tops (Natural slope & extended width) ───────────────
    # Move OUTWARD slightly from shoulder joints to cover physical shoulders
    d["left_shoulder_dst"]  = (vl_s[0] - SW * 0.05, SY - SW * 0.05)
    d["right_shoulder_dst"] = (vr_s[0] + SW * 0.05, SY - SW * 0.05)
    
    # ── Chest Anchors ────────────────────────────────────────────────
    chest_y = SY + (hip_y - SY) * 0.35
    lx, rx = get_side_x(chest_y)
    # Chest anchors sit halfway between the center and the side edges
    d["left_chest"]   = (lx + (MX - lx) * 0.5, chest_y)
    d["right_chest"]  = (MX + (rx - MX) * 0.5, chest_y)
    d["center_chest"] = (MX, chest_y)

    # ── Constraint Anchors (Anti-Fold) ───────────────────────────────
    armpit_y = SY + (chest_y - SY) * 0.5
    lx, rx = get_side_x(armpit_y)
    # Armpits sit slightly inward from the outer body line
    d["left_armpit"]  = (lx + SW * 0.05, armpit_y)
    d["right_armpit"] = (rx - SW * 0.05, armpit_y)
    d["upper_chest_center"] = (MX, neck_y + SW * 0.25)

    upper_side_y = chest_y + (hip_y - chest_y) * 0.15
    lx, rx = get_side_x(upper_side_y)
    d["upper_side_left"]  = (lx, upper_side_y)
    d["upper_side_right"] = (rx, upper_side_y)

    # ── Mid-Edge Anchors (Distributes tension) ───────────────────────
    waist_y = hip_y - SW * 0.10
    dy = (waist_y - upper_side_y) / 3.0
    lx1, rx1 = get_side_x(upper_side_y + dy)
    lx2, rx2 = get_side_x(upper_side_y + 2*dy)
    d["mid_side_left_1"]  = (lx1, upper_side_y + dy)
    d["mid_side_left_2"]  = (lx2, upper_side_y + 2*dy)
    d["mid_side_right_1"] = (rx1, upper_side_y + dy)
    d["mid_side_right_2"] = (rx2, upper_side_y + 2*dy)

    # ── Sleeve Tips ──────────────────────────────────────────────────
    # Sleeves extend OUTWARD from shoulders
    d["left_sleeve_dst"]  = (vl_s[0] - SW * 0.20, SY + SW * 0.15)
    d["right_sleeve_dst"] = (vr_s[0] + SW * 0.20, SY + SW * 0.15)
    
    d["left_sleeve_mid"]  = (vl_s[0] - SW * 0.10, SY + SW * 0.05)
    d["right_sleeve_mid"] = (vr_s[0] + SW * 0.10, SY + SW * 0.05)

    # ── Sleeve Ends (for long sleeves) ───────────────────────────────
    if vl_e:
        sx = vl_s[0] + (vl_e[0] - vl_s[0]) * 0.40
        sy = vl_s[1] + (vl_e[1] - vl_s[1]) * 0.40
        # Nudge outward from line
        d["left_sleeve_end"] = (sx - SW * 0.10, sy)
    else:
        d["left_sleeve_end"] = (vl_s[0] - SW * 0.35, vl_s[1] + SW * 0.25)

    if vr_e:
        sx = vr_s[0] + (vr_e[0] - vr_s[0]) * 0.40
        sy = vr_s[1] + (vr_e[1] - vr_s[1]) * 0.40
        # Nudge outward from line
        d["right_sleeve_end"] = (sx + SW * 0.10, sy)
    else:
        d["right_sleeve_end"] = (vr_s[0] + SW * 0.35, vr_s[1] + SW * 0.25)

    # ── Elbows & Cuffs ───────────────────────────────────────────────
    d["left_elbow"]  = vl_e if vl_e else (vl_s[0] - SW * 0.15, vl_s[1] + SW * 0.77)
    d["right_elbow"] = vr_e if vr_e else (vr_s[0] + SW * 0.15, vr_s[1] + SW * 0.77)
    d["left_cuff"]   = vl_w if vl_w else (vl_s[0] - SW * 0.18, vl_s[1] + SW * 1.50)
    d["right_cuff"]  = vr_w if vr_w else (vr_s[0] + SW * 0.18, vr_s[1] + SW * 1.50)

    # ── Lapels ───────────────────────────────────────────────────────
    d["left_lapel"]  = (MX - SW * 0.20, SY + SW * 0.35)
    d["right_lapel"] = (MX + SW * 0.20, SY + SW * 0.35)

    # ── Waist / Hem (Curve & Contour) ────────────────────────────────
    lx, rx = get_side_x(hip_y - SW * 0.10)
    d["left_side_waist"]  = (lx, hip_y - SW * 0.10)
    d["right_side_waist"] = (rx, hip_y - SW * 0.10)

    # Hem: Matches hip width visually, center drops for vertical curve
    lx, rx = get_side_x(hip_y)
    d["left_hem_ref"]  = (lx, hip_y)
    d["right_hem_ref"] = (rx, hip_y)
    d["center_hem"]    = (MX, hip_y + SW * 0.05)

    # ── Lower Body Specific (Knees & Ankles) ─────────────────────────
    if vl_k: d["left_knee"] = vl_k
    else: d["left_knee"] = (vl_h[0] if vl_h else lx, hip_y + SW * 1.2)
    
    if vr_k: d["right_knee"] = vr_k
    else: d["right_knee"] = (vr_h[0] if vr_h else rx, hip_y + SW * 1.2)
    
    if vl_a: d["left_ankle"] = vl_a
    else: d["left_ankle"] = (vl_h[0] if vl_h else lx, hip_y + SW * 2.5)

    if vr_a: d["right_ankle"] = vr_a
    else: d["right_ankle"] = (vr_h[0] if vr_h else rx, hip_y + SW * 2.5)

    # Compute inner and outer points for legs to give pants physical width!
    # A pant leg is roughly 25% of shoulder width at knee, 20% at ankle.
    kw = SW * 0.13  # half-width at knee
    aw = SW * 0.11  # half-width at ankle
    
    # Left Leg (Viewer-Left)
    d["left_knee_inner_dst"] = (d["left_knee"][0] + kw, d["left_knee"][1])
    d["left_knee_outer_dst"] = (d["left_knee"][0] - kw, d["left_knee"][1])
    d["left_ankle_inner_dst"] = (d["left_ankle"][0] + aw, d["left_ankle"][1])
    d["left_ankle_outer_dst"] = (d["left_ankle"][0] - aw, d["left_ankle"][1])
    
    # Right Leg (Viewer-Right)
    d["right_knee_inner_dst"] = (d["right_knee"][0] - kw, d["right_knee"][1])
    d["right_knee_outer_dst"] = (d["right_knee"][0] + kw, d["right_knee"][1])
    d["right_ankle_inner_dst"] = (d["right_ankle"][0] - aw, d["right_ankle"][1])
    d["right_ankle_outer_dst"] = (d["right_ankle"][0] + aw, d["right_ankle"][1])
    
    return d


# ---------------------------------------------------------------------------
# Garment schemas — anchor UVs match standard flat-lay garment proportions
# ---------------------------------------------------------------------------

TSHIRT_SCHEMA = GarmentKeypointSchema(
    category="tshirt",
    anchors=[
        # Collar curve (HIGH weight) + Collar Lock (HIGH weight)
        GarmentAnchor("collar_left",      (0.38, 0.05), "collar_left", weight=3),
        GarmentAnchor("collar_bottom",    (0.50, 0.12), "collar_bottom", weight=3),
        GarmentAnchor("collar_right",     (0.62, 0.05), "collar_right", weight=3),
        GarmentAnchor("upper_chest_center", (0.50, 0.20), "upper_chest_center", weight=3),
        
        # Shoulder seams (HIGH weight)
        GarmentAnchor("left_shoulder",    (0.18, 0.15), "left_shoulder_dst", weight=3),
        GarmentAnchor("right_shoulder",   (0.82, 0.15), "right_shoulder_dst", weight=3),
        
        # Armpit Anti-Fold Constraint (LOW weight)
        GarmentAnchor("left_armpit",      (0.20, 0.25), "left_armpit", weight=1),
        GarmentAnchor("right_armpit",     (0.80, 0.25), "right_armpit", weight=1),
        
        # Chest anchors (LOW weight - prevents flat sticker look)
        GarmentAnchor("left_chest",       (0.25, 0.40), "left_chest", weight=1),
        GarmentAnchor("center_chest",     (0.50, 0.40), "center_chest", weight=1),
        GarmentAnchor("right_chest",      (0.75, 0.40), "right_chest", weight=1),

        # Side edges & Waist (MEDIUM weight - structural tension)
        GarmentAnchor("upper_side_left",  (0.20, 0.35), "upper_side_left", weight=2),
        GarmentAnchor("upper_side_right", (0.80, 0.35), "upper_side_right", weight=2),
        GarmentAnchor("mid_side_left_1",  (0.20, 0.45), "mid_side_left_1", weight=2),
        GarmentAnchor("mid_side_right_1", (0.80, 0.45), "mid_side_right_1", weight=2),
        GarmentAnchor("mid_side_left_2",  (0.20, 0.55), "mid_side_left_2", weight=2),
        GarmentAnchor("mid_side_right_2", (0.80, 0.55), "mid_side_right_2", weight=2),
        GarmentAnchor("left_side_waist",  (0.20, 0.65), "left_side_waist", weight=2),
        GarmentAnchor("right_side_waist", (0.80, 0.65), "right_side_waist", weight=2),
        
        # Sleeve tips (LOW weight)
        GarmentAnchor("left_sleeve_end",  (0.05, 0.32), "left_sleeve_dst", weight=1),
        GarmentAnchor("left_sleeve_mid",  (0.12, 0.25), "left_sleeve_mid", weight=1),
        GarmentAnchor("right_sleeve_end", (0.95, 0.32), "right_sleeve_dst", weight=1),
        GarmentAnchor("right_sleeve_mid", (0.88, 0.25), "right_sleeve_mid", weight=1),
        
        # Hem corners & center curve (MEDIUM weight)
        GarmentAnchor("left_hem",         (0.15, 0.94), "left_hem_ref", weight=2),
        GarmentAnchor("center_hem",       (0.50, 0.97), "center_hem", weight=2),
        GarmentAnchor("right_hem",        (0.85, 0.94), "right_hem_ref", weight=2),
    ]
)

SHIRT_SCHEMA = GarmentKeypointSchema(
    category="shirt",
    anchors=[
        # Collar and Neck
        GarmentAnchor("collar_center",    (0.50, 0.06), "collar_bottom", weight=3),
        GarmentAnchor("collar_left",      (0.40, 0.06), "collar_left", weight=3),
        GarmentAnchor("collar_right",     (0.60, 0.06), "collar_right", weight=3),
        
        # Shoulders
        GarmentAnchor("left_shoulder",    (0.20, 0.12), "left_shoulder_dst", weight=3),
        GarmentAnchor("right_shoulder",   (0.80, 0.12), "right_shoulder_dst", weight=3),
        
        # Anti-Fold Chest and Armpits
        GarmentAnchor("left_armpit",      (0.22, 0.25), "left_armpit", weight=1),
        GarmentAnchor("right_armpit",     (0.78, 0.25), "right_armpit", weight=1),
        GarmentAnchor("left_chest",       (0.25, 0.35), "left_chest", weight=1),
        GarmentAnchor("right_chest",      (0.75, 0.35), "right_chest", weight=1),
        GarmentAnchor("center_chest",     (0.50, 0.35), "center_chest", weight=1),
        
        # Sleeves (Long)
        GarmentAnchor("left_elbow",       (0.08, 0.42), "left_elbow", weight=2),
        GarmentAnchor("right_elbow",      (0.92, 0.42), "right_elbow", weight=2),
        GarmentAnchor("left_cuff",        (0.05, 0.72), "left_cuff", weight=2),
        GarmentAnchor("right_cuff",       (0.95, 0.72), "right_cuff", weight=2),
        
        # Sides
        GarmentAnchor("upper_side_left",  (0.20, 0.35), "upper_side_left", weight=2),
        GarmentAnchor("upper_side_right", (0.80, 0.35), "upper_side_right", weight=2),
        GarmentAnchor("mid_side_left",    (0.19, 0.50), "mid_side_left_1", weight=2),
        GarmentAnchor("mid_side_right",   (0.81, 0.50), "mid_side_right_1", weight=2),
        GarmentAnchor("left_side_waist",  (0.18, 0.63), "left_side_waist", weight=2),
        GarmentAnchor("right_side_waist", (0.82, 0.63), "right_side_waist", weight=2),
        
        # Hem
        GarmentAnchor("left_hem",         (0.20, 0.93), "left_hem_ref", weight=2),
        GarmentAnchor("right_hem",        (0.80, 0.93), "right_hem_ref", weight=2),
        GarmentAnchor("center_hem",       (0.50, 0.95), "center_hem", weight=2),
    ]
)

JACKET_SCHEMA = GarmentKeypointSchema(
    category="jacket",
    anchors=[
        # Collar and Lapels
        GarmentAnchor("collar_left",      (0.44, 0.07), "collar_left", weight=3),
        GarmentAnchor("collar_right",     (0.56, 0.07), "collar_right", weight=3),
        GarmentAnchor("collar_bottom",    (0.50, 0.12), "collar_bottom", weight=3),
        GarmentAnchor("left_lapel",       (0.38, 0.22), "left_lapel", weight=2),
        GarmentAnchor("right_lapel",      (0.62, 0.22), "right_lapel", weight=2),
        
        # Shoulders
        GarmentAnchor("left_shoulder",    (0.17, 0.12), "left_shoulder_dst", weight=3),
        GarmentAnchor("right_shoulder",   (0.83, 0.12), "right_shoulder_dst", weight=3),
        
        # Chest and Armpits
        GarmentAnchor("left_armpit",      (0.20, 0.25), "left_armpit", weight=1),
        GarmentAnchor("right_armpit",     (0.80, 0.25), "right_armpit", weight=1),
        GarmentAnchor("left_chest",       (0.25, 0.35), "left_chest", weight=1),
        GarmentAnchor("right_chest",      (0.75, 0.35), "right_chest", weight=1),
        
        # Sleeves (Long)
        GarmentAnchor("left_elbow",       (0.07, 0.44), "left_elbow", weight=2),
        GarmentAnchor("right_elbow",      (0.93, 0.44), "right_elbow", weight=2),
        GarmentAnchor("left_cuff",        (0.04, 0.72), "left_cuff", weight=2),
        GarmentAnchor("right_cuff",       (0.96, 0.72), "right_cuff", weight=2),
        
        # Sides
        GarmentAnchor("upper_side_left",  (0.18, 0.35), "upper_side_left", weight=2),
        GarmentAnchor("upper_side_right", (0.82, 0.35), "upper_side_right", weight=2),
        GarmentAnchor("mid_side_left",    (0.17, 0.50), "mid_side_left_1", weight=2),
        GarmentAnchor("mid_side_right",   (0.83, 0.50), "mid_side_right_1", weight=2),
        GarmentAnchor("left_side_waist",  (0.16, 0.62), "left_side_waist", weight=2),
        GarmentAnchor("right_side_waist", (0.84, 0.62), "right_side_waist", weight=2),
        
        # Hem
        GarmentAnchor("left_hem",         (0.18, 0.93), "left_hem_ref", weight=2),
        GarmentAnchor("right_hem",        (0.82, 0.93), "right_hem_ref", weight=2),
        GarmentAnchor("center_hem",       (0.50, 0.95), "center_hem", weight=2),
    ]
)

LOWER_BODY_SCHEMA = GarmentKeypointSchema(
    category="lower_body",
    anchors=[
        # Waist
        GarmentAnchor("waist_left",       (0.20, 0.05), "left_side_waist", weight=3),
        GarmentAnchor("waist_right",      (0.80, 0.05), "right_side_waist", weight=3),
        GarmentAnchor("waist_center",     (0.50, 0.08), "center_hem", weight=3),
        
        # Hips (Structural width)
        GarmentAnchor("hip_left",         (0.15, 0.15), "left_hem_ref", weight=3),
        GarmentAnchor("hip_right",        (0.85, 0.15), "right_hem_ref", weight=3),
        
        # Knees
        GarmentAnchor("left_knee_inner",  (0.40, 0.50), "left_knee_inner", weight=2),
        GarmentAnchor("left_knee_outer",  (0.10, 0.50), "left_knee_outer", weight=2),
        GarmentAnchor("right_knee_inner", (0.60, 0.50), "right_knee_inner", weight=2),
        GarmentAnchor("right_knee_outer", (0.90, 0.50), "right_knee_outer", weight=2),
        
        # Ankles / Hem
        GarmentAnchor("left_ankle_inner", (0.40, 0.95), "left_ankle_inner", weight=2),
        GarmentAnchor("left_ankle_outer", (0.10, 0.95), "left_ankle_outer", weight=2),
        GarmentAnchor("right_ankle_inner",(0.60, 0.95), "right_ankle_inner", weight=2),
        GarmentAnchor("right_ankle_outer",(0.90, 0.95), "right_ankle_outer", weight=2),
    ]
)


# ---------------------------------------------------------------------------
# Registry and public API
# ---------------------------------------------------------------------------

_SCHEMA_REGISTRY: Dict[str, GarmentKeypointSchema] = {
    "tshirt":  TSHIRT_SCHEMA,
    "shirt":   SHIRT_SCHEMA,
    "jacket":  JACKET_SCHEMA,
    "t-shirt": TSHIRT_SCHEMA,
    "t_shirt": TSHIRT_SCHEMA,
    "tops":    TSHIRT_SCHEMA,
    "lower body": LOWER_BODY_SCHEMA,
    "lower_body": LOWER_BODY_SCHEMA,
    "pants":   LOWER_BODY_SCHEMA,
    "skirt":   LOWER_BODY_SCHEMA,
    "shorts":  LOWER_BODY_SCHEMA,
}


def get_garment_schema(category: str) -> GarmentKeypointSchema:
    key = category.lower().strip()
    if key not in _SCHEMA_REGISTRY:
        raise ValueError(
            f"Unknown garment category: '{category}'. "
            f"Supported: {list_supported_categories()}"
        )
    return _SCHEMA_REGISTRY[key]


def list_supported_categories() -> List[str]:
    return ["tshirt", "shirt", "jacket", "lower_body", "pants", "skirt"]


def resolve_points(
    schema: GarmentKeypointSchema,
    garment_image: np.ndarray,
    keypoints: list,
    person_w: int,
    person_h: int,
    shoulder_scale: float = 1.0,
    garment_mask: np.ndarray | None = None,
    person_mask: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    g_h, g_w = garment_image.shape[:2]
    src_pts  = schema.get_src_points(g_w, g_h, garment_mask)
    dst_pts  = schema.get_dst_points(keypoints, person_w, person_h, shoulder_scale, person_mask)
    if dst_pts is None:
        return None, None
    return src_pts, dst_pts