"""Tests for Size Recommendation"""

import pytest
from src.models import Measurements
from src.size_recommendation import (
    recommend_size,
    calculate_fit_score,
    calculate_measurement_fit,
    get_size_alternatives,
    explain_recommendation,
    compare_sizes,
    find_closest_size,
)


class TestRecommendSize:
    """Test size recommendation"""
    
    @pytest.fixture
    def user_measurements(self):
        """Create test measurements"""
        return Measurements(
            shoulder_width_cm=40.0,
            chest_circumference_cm=85.0,
            torso_length_cm=61.0,
            source="inferred",
            confidence=0.85
        )
    
    @pytest.fixture
    def size_chart(self):
        """Create test size chart"""
        return {
            "XS": {"shoulder_width_cm": 35.0, "chest_circumference_cm": 75.0, "torso_length_cm": 55.0},
            "S": {"shoulder_width_cm": 37.0, "chest_circumference_cm": 80.0, "torso_length_cm": 58.0},
            "M": {"shoulder_width_cm": 40.0, "chest_circumference_cm": 85.0, "torso_length_cm": 61.0},
            "L": {"shoulder_width_cm": 43.0, "chest_circumference_cm": 90.0, "torso_length_cm": 64.0},
            "XL": {"shoulder_width_cm": 46.0, "chest_circumference_cm": 95.0, "torso_length_cm": 67.0},
        }
    
    def test_recommend_perfect_fit(self, user_measurements, size_chart):
        """Test recommendation for perfect fit"""
        recommendation = recommend_size(user_measurements, size_chart)
        
        assert recommendation.size == "M"
        assert recommendation.confidence > 0.9
    
    def test_recommend_close_fit(self, size_chart):
        """Test recommendation for close fit"""
        measurements = Measurements(
            shoulder_width_cm=41.0,
            chest_circumference_cm=86.0,
            torso_length_cm=62.0,
            source="inferred"
        )
        
        recommendation = recommend_size(measurements, size_chart)
        assert recommendation.size in ["M", "L"]
    
    def test_recommend_empty_chart(self, user_measurements):
        """Test with empty size chart"""
        with pytest.raises(ValueError):
            recommend_size(user_measurements, {})
    
    def test_recommended_sizes(self, user_measurements, size_chart):
        """Test that recommended sizes are returned"""
        recommendation = recommend_size(user_measurements, size_chart)
        
        assert isinstance(recommendation.recommended_sizes, list)
        assert len(recommendation.recommended_sizes) > 0
    
    def test_fit_scores(self, user_measurements, size_chart):
        """Test that fit scores are calculated"""
        recommendation = recommend_size(user_measurements, size_chart)
        
        assert isinstance(recommendation.fit_scores, dict)
        assert len(recommendation.fit_scores) == len(size_chart)


class TestMeasurementFit:
    """Test measurement fit calculation"""
    
    def test_perfect_fit(self):
        """Test perfect fit (0% difference)"""
        score = calculate_measurement_fit(100.0, 100.0)
        assert score == 1.0
    
    def test_small_difference(self):
        """Test small difference"""
        score = calculate_measurement_fit(100.0, 105.0)  # 5% difference
        assert 0.5 < score < 1.0
    
    def test_large_difference(self):
        """Test large difference"""
        score = calculate_measurement_fit(100.0, 150.0)  # 50% difference
        assert score == 0.0
    
    def test_zero_size_value(self):
        """Test with zero size value"""
        score = calculate_measurement_fit(100.0, 0.0)
        assert score == 0.0


class TestFitScore:
    """Test overall fit score"""
    
    def test_fit_score_calculation(self):
        """Test fit score calculation"""
        user_measurements = Measurements(
            shoulder_width_cm=40.0,
            chest_circumference_cm=85.0,
            torso_length_cm=61.0,
            source="inferred"
        )
        
        size_measurements = {
            "shoulder_width_cm": 40.0,
            "chest_circumference_cm": 85.0,
            "torso_length_cm": 61.0
        }
        
        score = calculate_fit_score(user_measurements, size_measurements)
        
        assert 0.9 < score <= 1.0
    
    def test_fit_score_partial_measurements(self):
        """Test fit score with missing measurements"""
        user_measurements = Measurements(
            shoulder_width_cm=40.0,
            chest_circumference_cm=85.0,
            torso_length_cm=61.0,
            source="inferred"
        )
        
        size_measurements = {
            "shoulder_width_cm": 40.0,
        }
        
        score = calculate_fit_score(user_measurements, size_measurements)
        
        assert 0 <= score <= 1


class TestSizeAlternatives:
    """Test alternative size suggestions"""
    
    def test_get_alternatives(self):
        """Test getting alternative sizes"""
        fit_scores = {
            "XS": 0.5,
            "S": 0.75,
            "M": 0.95,
            "L": 0.7,
            "XL": 0.3,
        }
        
        alternatives = get_size_alternatives(fit_scores, top_n=3)
        
        assert len(alternatives) == 3
        assert alternatives[0] == "M"
        assert alternatives[1] == "S"


class TestExplainRecommendation:
    """Test recommendation explanation"""
    
    def test_explain_recommendation(self):
        """Test generating explanation"""
        from src.models import SizeRecommendation
        
        measurements = Measurements(
            shoulder_width_cm=40.0,
            chest_circumference_cm=85.0,
            torso_length_cm=61.0,
            source="inferred"
        )
        
        recommendation = SizeRecommendation(
            size="M",
            confidence=0.95,
            fit_scores={"M": 0.95},
            recommended_sizes=["M", "L"]
        )
        
        explanation = explain_recommendation(recommendation, measurements)
        
        assert "Size Recommendation: M" in explanation
        assert "Confidence" in explanation
        assert "Your Measurements" in explanation


class TestCompareSizes:
    """Test size comparison"""
    
    def test_compare_sizes(self):
        """Test comparing two sizes"""
        size_chart = {
            "M": {"shoulder_width_cm": 40.0, "chest_circumference_cm": 85.0},
            "L": {"shoulder_width_cm": 43.0, "chest_circumference_cm": 90.0},
        }
        
        differences = compare_sizes("M", "L", size_chart)
        
        assert differences["shoulder_width_cm"] == 3.0
        assert differences["chest_circumference_cm"] == 5.0
    
    def test_compare_invalid_size(self):
        """Test comparing with invalid size"""
        size_chart = {
            "M": {"shoulder_width_cm": 40.0},
        }
        
        with pytest.raises(ValueError):
            compare_sizes("M", "INVALID", size_chart)


class TestFindClosestSize:
    """Test finding closest size"""
    
    def test_find_closest(self):
        """Test finding closest size"""
        measurements = Measurements(
            shoulder_width_cm=40.5,
            chest_circumference_cm=85.5,
            torso_length_cm=61.0,
            source="inferred"
        )
        
        size_chart = {
            "S": {"shoulder_width_cm": 37.0, "chest_circumference_cm": 80.0, "torso_length_cm": 58.0},
            "M": {"shoulder_width_cm": 40.0, "chest_circumference_cm": 85.0, "torso_length_cm": 61.0},
            "L": {"shoulder_width_cm": 43.0, "chest_circumference_cm": 90.0, "torso_length_cm": 64.0},
        }
        
        closest = find_closest_size(measurements, size_chart)
        
        assert closest == "M"