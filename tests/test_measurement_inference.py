"""Measurement Inference Module - FIXED with Real Calibration"""

from typing import Tuple
import numpy as np

from src.models import Measurements, SegmentationResult, PoseResult
from src.pose_detection import calculate_torso_length, get_keypoint_coordinate


# ============================================================================
# CALIBRATION SETTINGS
# ============================================================================

# THIS IS THE KEY FIX!
# Calibrated pixels-per-cm ratio based on standard images
# Updated from 10.0 to 7.0 based on real MediaPipe measurements

PIXELS_PER_CM = 7.0  # FIXED: Was 10.0, now 7.0

# Shoulder width to chest circumference multiplier
# Empirically determined ratio (chest is ~2.3x shoulder width on average)
SHOULDER_TO_CIRCUMFERENCE = 2.3


def infer_measurements(
    pose_result: PoseResult,
    seg_result: SegmentationResult
) -> Measurements:
    """
    Infer body measurements from pose and segmentation
    
    Args:
        pose_result: Pose detection result with keypoints
        seg_result: Segmentation result with body mask
        
    Returns:
        Measurements with shoulder, chest, and torso length
    """
    if not pose_result or not seg_result:
        raise ValueError("Both pose and segmentation results required")
    
    # Calculate measurements in pixels
    shoulder_width_px = pose_result.shoulder_width_px
    torso_length_px = calculate_torso_length(pose_result.keypoints)
    chest_width_px = _estimate_chest_width(pose_result)
    
    # Convert pixels to centimeters
    # Using calibrated pixel-to-cm ratio
    shoulder_width_cm = shoulder_width_px / PIXELS_PER_CM
    torso_length_cm = torso_length_px / PIXELS_PER_CM
    chest_circumference_cm = chest_width_px / PIXELS_PER_CM * SHOULDER_TO_CIRCUMFERENCE
    
    # Calculate confidence based on detection quality
    confidence = _calculate_confidence(pose_result, shoulder_width_cm, chest_circumference_cm)
    
    measurements = Measurements(
        shoulder_width_cm=round(shoulder_width_cm, 2),
        chest_circumference_cm=round(chest_circumference_cm, 2),
        torso_length_cm=round(torso_length_cm, 2),
        source='inferred',
        confidence=round(confidence, 2)
    )
    
    return measurements


def _estimate_chest_width(pose_result: PoseResult) -> float:
    """
    Estimate chest width from shoulder width
    
    Heuristic: Chest width is approximately 1.1-1.2x shoulder width
    (accounting for rib cage)
    
    Args:
        pose_result: Pose detection result
        
    Returns:
        Estimated chest width in pixels
    """
    shoulder_width = pose_result.shoulder_width_px
    
    # Chest is wider than shoulders due to rib cage
    chest_width = shoulder_width * 1.15
    
    return chest_width


def _calculate_confidence(
    pose_result: PoseResult,
    shoulder_width_cm: float,
    chest_circumference_cm: float
) -> float:
    """
    Calculate measurement confidence
    
    Based on:
    1. Number of keypoints detected
    2. If pose is frontal
    3. If measurements are in reasonable range
    
    Args:
        pose_result: Pose detection result
        shoulder_width_cm: Calculated shoulder width
        chest_circumference_cm: Calculated chest circumference
        
    Returns:
        Confidence score (0-1)
    """
    confidence = 0.5
    
    # More keypoints = higher confidence
    keypoint_bonus = min(len(pose_result.keypoints) / 20.0, 0.3)
    confidence += keypoint_bonus
    
    # Frontal pose = higher confidence
    if pose_result.is_frontal:
        confidence += 0.1
    
    # Check if measurements are reasonable
    # Adult shoulders typically 35-50 cm
    if 35 <= shoulder_width_cm <= 50:
        confidence += 0.05
    
    # Adult chest typically 75-120 cm
    if 75 <= chest_circumference_cm <= 120:
        confidence += 0.05
    
    # Cap at 0.95 (always leave room for uncertainty)
    confidence = min(confidence, 0.95)
    
    return confidence


def validate_measurements(measurements: Measurements) -> Tuple[bool, str]:
    """
    Validate that measurements are in reasonable range
    
    Args:
        measurements: Measurements to validate
        
    Returns:
        (is_valid, error_message) tuple
    """
    errors = []
    
    # Shoulder width: typical adult range 30-55 cm
    if not (30 <= measurements.shoulder_width_cm <= 55):
        errors.append(
            f"Invalid shoulder width: {measurements.shoulder_width_cm}cm "
            f"(expected 30-55cm)"
        )
    
    # Chest circumference: typical adult range 70-130 cm
    if not (70 <= measurements.chest_circumference_cm <= 130):
        errors.append(
            f"Invalid chest circumference: {measurements.chest_circumference_cm}cm "
            f"(expected 70-130cm)"
        )
    
    # Torso length: typical adult range 40-80 cm
    if not (40 <= measurements.torso_length_cm <= 80):
        errors.append(
            f"Invalid torso length: {measurements.torso_length_cm}cm "
            f"(expected 40-80cm)"
        )
    
    is_valid = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    
    return is_valid, error_message


def recalibrate_pixels_per_cm(reference_width_cm: float, measured_width_px: float) -> float:
    """
    Recalibrate pixels-per-cm ratio based on reference measurement
    
    Usage: If user provides a reference object (hand, credit card),
    we can calibrate the exact pixel-to-cm ratio for that image
    
    Args:
        reference_width_cm: Known width in cm (e.g., 8.5 for credit card)
        measured_width_px: Measured width in pixels from image
        
    Returns:
        New pixels-per-cm ratio
    """
    new_ratio = measured_width_px / reference_width_cm
    return new_ratio


# ============================================================================
# DEBUG UTILITIES
# ============================================================================

def print_measurement_debug_info(pose_result: PoseResult, measurements: Measurements):
    """Print debug information about measurements"""
    print("\n" + "="*70)
    print("MEASUREMENT DEBUG INFO")
    print("="*70)
    print(f"Shoulder Width (px): {pose_result.shoulder_width_px:.2f}")
    print(f"Shoulder Width (cm): {measurements.shoulder_width_cm:.2f}")
    print(f"Pixels per cm: {PIXELS_PER_CM}")
    print(f"\nChest Circumference (cm): {measurements.chest_circumference_cm:.2f}")
    print(f"Torso Length (cm): {measurements.torso_length_cm:.2f}")
    print(f"Confidence: {measurements.confidence * 100:.1f}%")
    print("="*70 + "\n")