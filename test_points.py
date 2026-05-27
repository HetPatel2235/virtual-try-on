import cv2
import numpy as np
import sys

from ml_ai.core.garment_keypoints import get_garment_schema

def test():
    schema = get_garment_schema("shirt")
    print(f"Total anchors: {len(schema.anchors)}")
    print(f"Expected weight sum: {sum(a.weight for a in schema.anchors if a.body_landmark)}")

    # Dummy keypoints
    from collections import namedtuple
    Keypoint = namedtuple('Keypoint', ['name', 'x', 'y', 'score'])
    kps = [
        Keypoint("left_shoulder", 300, 150, 0.9),
        Keypoint("right_shoulder", 100, 150, 0.9),
        Keypoint("left_hip", 300, 400, 0.9),
        Keypoint("right_hip", 100, 400, 0.9),
    ]

    # Dummy mask
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(mask, (80, 150), (320, 400), 255, -1)

    pts = schema.get_dst_points(kps, 500, 500, shoulder_scale=1.0, person_mask=mask)
    if pts is None:
        print("get_dst_points returned None!")
    else:
        print(f"get_dst_points returned {len(pts)} points.")

if __name__ == "__main__":
    test()
