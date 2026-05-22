"""Garment Warping and Scaling Module"""

from typing import Tuple, Optional
import numpy as np
import cv2


def scale_garment(
    garment_image: np.ndarray,
    target_width: int,
    target_height: int
) -> np.ndarray:
    """
    Scale garment to target dimensions
    
    Args:
        garment_image: Original garment image
        target_width: Target width in pixels
        target_height: Target height in pixels
        
    Returns:
        Scaled garment image
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")
    
    scaled = cv2.resize(garment_image, (target_width, target_height))
    return scaled


def rotate_garment(
    garment_image: np.ndarray,
    angle: float
) -> np.ndarray:
    """
    Rotate garment by specified angle
    
    Args:
        garment_image: Garment image
        angle: Rotation angle in degrees
        
    Returns:
        Rotated garment image
    """
    height, width = garment_image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(
        garment_image,
        rotation_matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rotated


def adjust_garment_fit(
    garment_image: np.ndarray,
    user_shoulder_width_cm: float,
    garment_shoulder_width_cm: float,
    pixels_per_cm: float = 10.0
) -> np.ndarray:
    """
    Adjust garment to fit user measurements
    
    Args:
        garment_image: Garment image
        user_shoulder_width_cm: User's shoulder width in cm
        garment_shoulder_width_cm: Garment's shoulder width in cm
        pixels_per_cm: Conversion factor
        
    Returns:
        Adjusted garment image
    """
    # Calculate scale factor
    user_shoulder_width_px = user_shoulder_width_cm * pixels_per_cm
    garment_shoulder_width_px = garment_shoulder_width_cm * pixels_per_cm
    
    if garment_shoulder_width_px <= 0:
        raise ValueError("Garment shoulder width must be positive")
    
    scale_factor = user_shoulder_width_px / garment_shoulder_width_px
    
    # Calculate new dimensions
    height, width = garment_image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Scale garment
    adjusted = scale_garment(garment_image, new_width, new_height)
    
    return adjusted


def create_garment_mask(garment_image: np.ndarray) -> np.ndarray:
    """
    Create transparency mask for garment
    
    Args:
        garment_image: Garment image
        
    Returns:
        Binary mask (1 where garment, 0 where background)
    """
    # Check if image has alpha channel
    if len(garment_image.shape) == 3 and garment_image.shape[2] == 4:
        # Use alpha channel
        alpha = garment_image[:, :, 3]
        mask = (alpha > 127).astype(np.uint8)
    else:
        # Convert to grayscale and threshold
        if len(garment_image.shape) == 3:
            gray = cv2.cvtColor(garment_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = garment_image
        
        # Threshold (white background = 0, garment = 1)
        _, mask = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY_INV)
    
    return mask


def estimate_garment_dimensions(
    garment_image: np.ndarray
) -> Tuple[float, float]:
    """
    Estimate garment dimensions from image
    
    Args:
        garment_image: Garment image
        
    Returns:
        Tuple of (estimated_width_cm, estimated_height_cm)
    """
    height, width = garment_image.shape[:2]
    
    # Assuming 512x512 image represents a large garment
    # Rough estimation: 512 pixels ≈ 50 cm
    pixels_per_cm = 512 / 50
    
    estimated_width = width / pixels_per_cm
    estimated_height = height / pixels_per_cm
    
    return estimated_width, estimated_height