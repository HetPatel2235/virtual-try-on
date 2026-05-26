"""Pose Detection Module - Updated with Real MediaPipe"""

from typing import List
import numpy as np

from src.models import PoseResult, Keypoint
from src.mediapipe_real import create_real_pose_detector


# Create pose detector (will be initialized once)
_pose_detector = None


def get_pose_detector():
    """Get or create pose detector (singleton)"""
    global _pose_detector
    if _pose_detector is None:
        _pose_detector = create_real_pose_detector()
    return _pose_detector


def detect_pose(image: np.ndarray, pose_model=None) -> PoseResult:
    """
    Detect pose in image using REAL MediaPipe
    
    Args:
        image: Input image (H x W x 3) in BGR format
        pose_model: Optional pre-loaded model (ignored, uses real MediaPipe)
        
    Returns:
        PoseResult with keypoints, shoulder width, and pose info
        
    Raises:
        RuntimeError: If no pose detected in image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be numpy array")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be (H x W x 3)")
    
    # Get real pose detector
    detector = get_pose_detector()
    
    # Detect pose
    pose_result = detector.detect_pose(image)
    
    # Validate detection
    if not check_critical_keypoints(pose_result.keypoints):
        raise RuntimeError(
            "Missing critical keypoints (shoulders or hips). "
            "Ensure person is facing camera with torso or lower body visible."
        )
    
    return pose_result


def check_critical_keypoints(keypoints: List[Keypoint]) -> bool:
    """
    Check if critical keypoints are present
    
    Args:
        keypoints: List of detected keypoints
        
    Returns:
        True if critical keypoints present, False otherwise
    """
    detected_names = {kp.name for kp in keypoints}
    has_shoulders = {'left_shoulder', 'right_shoulder'}.issubset(detected_names)
    has_hips = {'left_hip', 'right_hip'}.issubset(detected_names)
    
    return has_shoulders or has_hips


def validate_pose_quality(pose_result: PoseResult) -> tuple:
    """
    Validate pose quality
    
    Args:
        pose_result: PoseResult from detection
        
    Returns:
        (is_valid, errors) tuple
    """
    errors = []
    
    # Check if frontal
    if not pose_result.is_frontal:
        errors.append("Person is not directly facing camera")
    
    # Check keypoint count
    if len(pose_result.keypoints) < 8:
        errors.append(f"Too few keypoints detected: {len(pose_result.keypoints)}")
    
    # Check shoulder width
    if pose_result.shoulder_width_px < 15:
        errors.append("Shoulder width too small - person may be too far away")
    
    if pose_result.shoulder_width_px > 800:
        errors.append("Shoulder width too large - person may be too close")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_keypoint_coordinate(keypoints: List[Keypoint], keypoint_name: str) -> tuple:
    """
    Get pixel coordinates of a specific keypoint
    
    Args:
        keypoints: List of keypoints
        keypoint_name: Name of keypoint to find
        
    Returns:
        (x, y) tuple in pixels, or (0, 0) if not found
    """
    for kp in keypoints:
        if kp.name == keypoint_name:
            return (kp.x, kp.y)
    
    return (0, 0)


def calculate_torso_length(keypoints: List[Keypoint]) -> float:
    """
    Calculate torso length from keypoints
    
    Uses distance from neck area to hip area
    
    Args:
        keypoints: List of detected keypoints
        
    Returns:
        Torso length in pixels
    """
    # Find key points
    left_shoulder = next((kp for kp in keypoints if kp.name == 'left_shoulder'), None)
    right_shoulder = next((kp for kp in keypoints if kp.name == 'right_shoulder'), None)
    left_hip = next((kp for kp in keypoints if kp.name == 'left_hip'), None)
    right_hip = next((kp for kp in keypoints if kp.name == 'right_hip'), None)
    
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return 0.0
    
    # Calculate average shoulder y and hip y
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    
    # Torso length is vertical distance
    torso_length = abs(hip_y - shoulder_y)
    
    return torso_length


def calculate_inseam_length(keypoints: List[Keypoint]) -> float:
    """Calculate inseam length (crotch/hips to ankle) from keypoints."""
    left_hip = next((kp for kp in keypoints if kp.name == 'left_hip'), None)
    right_hip = next((kp for kp in keypoints if kp.name == 'right_hip'), None)
    left_ankle = next((kp for kp in keypoints if kp.name == 'left_ankle'), None)
    right_ankle = next((kp for kp in keypoints if kp.name == 'right_ankle'), None)
    
    if not (left_hip and right_hip):
        return 0.0
        
    hip_y = (left_hip.y + right_hip.y) / 2
    
    if left_ankle and right_ankle:
        ankle_y = (left_ankle.y + right_ankle.y) / 2
    elif left_ankle:
        ankle_y = left_ankle.y
    elif right_ankle:
        ankle_y = right_ankle.y
    else:
        return 0.0
        
    return abs(ankle_y - hip_y)


def calculate_hip_width(keypoints: List[Keypoint]) -> float:
    """Calculate hip width from keypoints."""
    left_hip = next((kp for kp in keypoints if kp.name == 'left_hip'), None)
    right_hip = next((kp for kp in keypoints if kp.name == 'right_hip'), None)
    
    if not (left_hip and right_hip):
        return 0.0
        
    return abs(right_hip.x - left_hip.x)