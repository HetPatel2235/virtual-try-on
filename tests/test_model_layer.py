"""Tests for Model Layer"""

import pytest
import numpy as np

from src.model_layer import (
    UNetSegmentationModel,
    DeepLabSegmentationModel,
    MediaPipePoseModel,
    OpenPosePoseModel,
    load_models,
)


class TestUNetSegmentation:
    """Test U-Net segmentation model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = UNetSegmentationModel()
        assert model is not None
        assert model.model_name == "UNet"
    
    def test_predict_valid_image(self):
        """Test prediction on valid image"""
        model = UNetSegmentationModel()
        image = np.ones((512, 512, 3), dtype=np.uint8)
        
        result = model.predict(image)
        
        assert result is not None
        assert result.mask is not None
        assert result.confidence > 0
    
    def test_predict_invalid_input(self):
        """Test prediction with invalid input"""
        model = UNetSegmentationModel()
        
        with pytest.raises(TypeError):
            model.predict([1, 2, 3])  # Not numpy array
    
    def test_torso_detection(self):
        """Test torso detection"""
        model = UNetSegmentationModel()
        image = np.ones((512, 512, 3), dtype=np.uint8)
        
        result = model.predict(image)
        
        assert result.torso_percentage > 0


class TestDeepLabSegmentation:
    """Test DeepLab segmentation model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = DeepLabSegmentationModel()
        assert model is not None
        assert model.model_name == "DeepLab"
    
    def test_predict_valid_image(self):
        """Test prediction on valid image"""
        model = DeepLabSegmentationModel()
        image = np.ones((512, 512, 3), dtype=np.uint8)
        
        result = model.predict(image)
        
        assert result is not None
        assert result.mask is not None
        assert result.confidence > 0
    
    def test_more_body_parts_than_unet(self):
        """Test DeepLab has more body parts"""
        unet = UNetSegmentationModel()
        deeplab = DeepLabSegmentationModel()
        
        image = np.ones((512, 512, 3), dtype=np.uint8)
        
        unet_result = unet.predict(image)
        deeplab_result = deeplab.predict(image)
        
        assert len(deeplab_result.body_parts) > len(unet_result.body_parts)


class TestMediaPipePose:
    """Test MediaPipe pose detection"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = MediaPipePoseModel()
        assert model is not None
        assert model.model_name == "MediaPipe"
    
    def test_predict_valid_image(self):
        """Test prediction on valid image"""
        model = MediaPipePoseModel()
        image = np.ones((512, 512, 3), dtype=np.uint8)
        
        result = model.predict(image)
        
        assert result is not None
        assert len(result.keypoints) > 0
        assert result.shoulder_width_px > 0
    
    def test_keypoint_detection(self):
        """Test keypoint detection"""
        model = MediaPipePoseModel()
        image = np.ones((512, 512, 3), dtype=np.uint8)
        
        result = model.predict(image)
        
        # Should detect shoulders
        shoulder_names = [k.name for k in result.keypoints]
        assert 'left_shoulder' in shoulder_names
        assert 'right_shoulder' in shoulder_names


class TestOpenPosePose:
    """Test OpenPose pose detection"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = OpenPosePoseModel()
        assert model is not None
        assert model.model_name == "OpenPose"
    
    def test_predict_valid_image(self):
        """Test prediction on valid image"""
        model = OpenPosePoseModel()
        image = np.ones((512, 512, 3), dtype=np.uint8)
        
        result = model.predict(image)
        
        assert result is not None
        assert len(result.keypoints) > 0
        assert result.shoulder_width_px > 0
    
    def test_more_keypoints_than_mediapipe(self):
        """Test OpenPose detects more keypoints"""
        mediapipe = MediaPipePoseModel()
        openpose = OpenPosePoseModel()
        
        image = np.ones((512, 512, 3), dtype=np.uint8)
        
        mp_result = mediapipe.predict(image)
        op_result = openpose.predict(image)
        
        assert len(op_result.keypoints) > len(mp_result.keypoints)


class TestLoadModels:
    """Test model loading"""
    
    def test_load_models(self):
        """Test loading models from config"""
        seg_model, pose_model = load_models("config/models.json")
        
        assert seg_model is not None
        assert pose_model is not None
        assert seg_model.model_name == "UNet"
        assert pose_model.model_name == "MediaPipe"
    
    def test_load_models_invalid_config(self):
        """Test loading with invalid config"""
        with pytest.raises(FileNotFoundError):
            load_models("nonexistent.json")