"""Body Segmentation Processing Module"""

import numpy as np
from typing import Dict, Optional

from src.models import SegmentationResult
from src.model_layer import SegmentationModel


def segment_body(
    image: np.ndarray,
    model: SegmentationModel,
    min_torso_percentage: float = 15.0,
    min_confidence: float = 0.6
) -> SegmentationResult:
    """
    Segment body in image
    
    Args:
        image: Input image (H x W x 3)
        model: Segmentation model instance
        min_torso_percentage: Minimum torso area percentage
        min_confidence: Minimum confidence threshold
        
    Returns:
        SegmentationResult with validation
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be numpy array")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be (H x W x 3)")
    
    # Run segmentation
    result = model.predict(image)
    
    # Validate segmentation quality
    is_valid, validation_warnings = validate_segmentation_quality(
        result,
        min_torso_percentage=min_torso_percentage,
        min_confidence=min_confidence
    )
    
    # Add validation warnings
    result.warnings.extend(validation_warnings)
    
    return result


def validate_segmentation_quality(
    result: SegmentationResult,
    min_torso_percentage: float = 15.0,
    min_confidence: float = 0.6
) -> tuple:
    """
    Validate segmentation quality
    
    Args:
        result: Segmentation result
        min_torso_percentage: Minimum torso area percentage
        min_confidence: Minimum confidence threshold
        
    Returns:
        Tuple of (is_valid, warnings)
    """
    warnings = []
    is_valid = True
    
    # Check torso percentage
    if result.torso_percentage < min_torso_percentage:
        warnings.append(
            f"Low torso area: {result.torso_percentage:.1f}% "
            f"(min: {min_torso_percentage}%)"
        )
        is_valid = False
    
    # Check confidence
    if result.confidence < min_confidence:
        warnings.append(
            f"Low confidence: {result.confidence:.2f} "
            f"(min: {min_confidence})"
        )
        is_valid = False
    
    return is_valid, warnings


def extract_body_part_mask(
    result: SegmentationResult,
    body_part: str
) -> Optional[np.ndarray]:
    """
    Extract mask for specific body part
    
    Args:
        result: Segmentation result
        body_part: Body part name (e.g., 'torso', 'left_arm')
        
    Returns:
        Binary mask for body part or None
    """
    if body_part not in result.body_parts:
        return None
    
    return result.body_parts[body_part]


def get_torso_region(result: SegmentationResult) -> Optional[Dict]:
    """
    Get bounding box of torso region
    
    Args:
        result: Segmentation result
        
    Returns:
        Dictionary with torso bounding box or None
    """
    torso_mask = extract_body_part_mask(result, 'torso')
    
    if torso_mask is None:
        return None
    
    # Find non-zero pixels
    coords = np.argwhere(torso_mask > 0)
    
    if len(coords) == 0:
        return None
    
    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return {
        'x_min': int(x_min),
        'y_min': int(y_min),
        'x_max': int(x_max),
        'y_max': int(y_max),
        'width': int(x_max - x_min),
        'height': int(y_max - y_min),
    }