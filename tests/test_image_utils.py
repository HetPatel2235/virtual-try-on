"""Tests for image utilities"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

from src.image_utils import (
    generate_unique_filename,
    load_image,
    save_image,
    get_image_dimensions,
    get_file_size,
)


class TestGenerateUniqueFilename:
    """Test filename generation"""
    
    def test_default_parameters(self):
        """Test with default parameters"""
        filename = generate_unique_filename()
        
        assert filename.startswith("image_")
        assert filename.endswith(".png")
    
    def test_custom_prefix_and_extension(self):
        """Test with custom prefix and extension"""
        filename = generate_unique_filename(prefix="photo", extension="jpg")
        
        assert filename.startswith("photo_")
        assert filename.endswith(".jpg")
    
    def test_uniqueness(self):
        """Test that filenames are unique"""
        names = [generate_unique_filename() for _ in range(10)]
        
        # All should be unique
        assert len(set(names)) == 10


class TestLoadImage:
    """Test image loading"""
    
    def test_load_nonexistent_image(self):
        """Test loading nonexistent image raises error"""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent.png")


class TestSaveImage:
    """Test image saving"""
    
    def test_save_creates_directories(self):
        """Test that save creates parent directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple image
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Save to nested path
            output_path = os.path.join(tmpdir, "subdir", "image.png")
            save_image(image, output_path)
            
            # Verify file exists
            assert os.path.exists(output_path)


class TestGetImageDimensions:
    """Test getting image dimensions"""
    
    def test_get_nonexistent_image(self):
        """Test getting dimensions of nonexistent image"""
        with pytest.raises(FileNotFoundError):
            get_image_dimensions("nonexistent.png")