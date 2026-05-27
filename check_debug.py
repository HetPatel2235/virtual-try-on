import cv2
import numpy as np

def check():
    warped_mask = cv2.imread("database/data/tryon_debug/debug_warped_mask.png", cv2.IMREAD_GRAYSCALE)
    collar = cv2.imread("database/data/tryon_debug/debug_collar_mask.png", cv2.IMREAD_GRAYSCALE)
    edge = cv2.imread("database/data/tryon_debug/debug_edge_shadow.png", cv2.IMREAD_GRAYSCALE)

    print(f"warped_mask max: {warped_mask.max() if warped_mask is not None else 'None'}")
    
    if collar is not None:
        print(f"collar max: {collar.max()}")
    
    if edge is not None:
        print(f"edge max: {edge.max()}")

check()
