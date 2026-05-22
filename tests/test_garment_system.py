"""Tests for Garment System"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.garment_manager import (
    GarmentManager,
    load_garment_metadata,
    list_available_garments,
)
from src.garment_warping import (
    scale_garment,
    rotate_garment,
    adjust_garment_fit,
    create_garment_mask,
)
from src.overlay import (
    overlay_garment,
    blend_images,
)


class TestGarmentManager:
    """Test garment manager"""
    
    def test_load_metadata(self):
        """Test loading metadata"""
        metadata = load_garment_metadata("tshirt-001")
        
        assert metadata is not None
        assert metadata['id'] == "tshirt-001"
        assert 'size_chart' in metadata
    
    def test_load_nonexistent_garment(self):
        """Test loading nonexistent garment"""
        with pytest.raises(FileNotFoundError):
            load_garment_metadata("nonexistent")
    
    def test_list_garments(self):
        """Test listing garments"""
        garments = list_available_garments()
        
        assert isinstance(garments, list)
        assert len(garments) > 0
        assert "tshirt-001" in garments


class TestGarmentWarping:
    """Test garment warping"""
    
    @pytest.fixture
    def sample_garment(self):
        """Create sample garment image"""
        return np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    def test_scale_garment(self, sample_garment):
        """Test scaling garment"""
        scaled = scale_garment(sample_garment, 256, 256)
        
        assert scaled.shape == (256, 256, 3)
    
    def test_scale_invalid_dimensions(self, sample_garment):
        """Test scaling with invalid dimensions"""
        with pytest.raises(ValueError):
            scale_garment(sample_garment, -1, 256)
    
    def test_rotate_garment(self, sample_garment):
        """Test rotating garment"""
        rotated = rotate_garment(sample_garment, 45)
        
        assert rotated.shape == sample_garment.shape
    
    def test_adjust_garment_fit(self, sample_garment):
        """Test adjusting garment fit"""
        adjusted = adjust_garment_fit(
            sample_garment,
            user_shoulder_width_cm=40,
            garment_shoulder_width_cm=40
        )
        
        assert adjusted is not None
    
    def test_create_mask(self, sample_garment):
        """Test creating garment mask"""
        mask = create_garment_mask(sample_garment)
        
        assert mask.shape[:2] == sample_garment.shape[:2]
        assert mask.max() <= 1


class TestOverlay:
    """Test overlay operations"""
    
    @pytest.fixture
    def background(self):
        """Create background image"""
        return np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    @pytest.fixture
    def garment(self):
        """Create garment image"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        return img
    
    def test_overlay_garment(self, background, garment):
        """Test overlaying garment"""
        result = overlay_garment(background, garment, (100, 100))
        
        assert result.shape == background.shape
        assert result is not background  # Different object
    
    def test_overlay_invalid_alpha(self, background, garment):
        """Test overlay with invalid alpha"""
        with pytest.raises(ValueError):
            overlay_garment(background, garment, (100, 100), alpha=1.5)
    
    def test_blend_images(self, background, garment):
        """Test blending images"""
        # Create same-size images for blending
        img1 = np.ones((512, 512, 3), dtype=np.uint8) * 100
        img2 = np.ones((512, 512, 3), dtype=np.uint8) * 200
        
        blended = blend_images(img1, img2, alpha=0.5)
        
        assert blended.shape == img1.shape