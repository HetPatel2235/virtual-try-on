import cv2
import numpy as np
from ml_ai.core.tryon_engine import TryOnEngine

def check():
    engine = TryOnEngine()
    person_img = cv2.imread("database/data/tryon_debug/debug_person.png")
    garment_img = cv2.imread("database/data/garments/shirt-001/garment.png", cv2.IMREAD_UNCHANGED)
    
    if person_img is None:
        print("Could not load person image!")
        return

    result = engine.run(person_img, garment_img, "shirt", blend_alpha=1.0, shoulder_scale=1.15)
    
    # Let's run segment_body to see what parts we get
    from ml_ai.core.segmentation import segment_body
    seg_result = segment_body(person_img, engine._seg_model)
    print("Parts:", list(seg_result.body_parts.keys()))
    cv2.imwrite("test_torso.png", seg_result.body_parts["torso"] * 255)
    cv2.imwrite("test_left_arm.png", seg_result.body_parts["left_arm"] * 255)
    cv2.imwrite("test_right_arm.png", seg_result.body_parts["right_arm"] * 255)
    
    if result.composite_image is None:
        print("TryOn failed:", result.error)
    else:
        cv2.imwrite("test_composite.png", result.composite_image)
        print("TryOn succeeded! Image written to test_composite.png")
        print("Warnings:", result.warnings)

check()
