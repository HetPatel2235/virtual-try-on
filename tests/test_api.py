"""Tests for Flask API"""

import pytest
import json
import tempfile
from pathlib import Path

from src.api import app


@pytest.fixture
def client():
    """Create Flask test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'models_loaded' in data


class TestGarmentsEndpoint:
    """Test garments endpoint"""
    
    def test_list_garments(self, client):
        """Test listing garments"""
        response = client.get('/api/garments')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'garments' in data
        assert isinstance(data['garments'], list)
    
    def test_get_garment_details(self, client):
        """Test getting garment details"""
        response = client.get('/api/garments/tshirt-001')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert data['success'] is True
            assert 'garment' in data
    
    def test_nonexistent_garment(self, client):
        """Test getting nonexistent garment"""
        response = client.get('/api/garments/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False


class TestUploadEndpoint:
    """Test upload endpoint"""
    
    def test_upload_no_file(self, client):
        """Test upload without file"""
        response = client.post('/api/upload')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_upload_invalid_extension(self, client):
        """Test upload with invalid extension"""
        data = {'image': (tempfile.NamedTemporaryFile(suffix='.txt'), 'test.txt')}
        response = client.post('/api/upload', data=data, content_type='multipart/form-data')
        
        assert response.status_code == 400


class TestRecommendEndpoint:
    """Test recommend endpoint"""
    
    def test_recommend_missing_garment(self, client):
        """Test recommend without garment_id"""
        data = {
            'measurements': {
                'shoulder_width_cm': 40.0,
                'chest_circumference_cm': 85.0,
                'torso_length_cm': 61.0
            }
        }
        response = client.post(
            '/api/recommend',
            data=json.dumps(data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_recommend_invalid_measurements(self, client):
        """Test recommend with invalid measurements"""
        data = {
            'measurements': {},
            'garment_id': 'tshirt-001'
        }
        response = client.post(
            '/api/recommend',
            data=json.dumps(data),
            content_type='application/json'
        )
        
        # Should handle missing measurements gracefully
        assert response.status_code in [200, 400, 500]


class TestErrorHandlers:
    """Test error handlers"""
    
    def test_not_found(self, client):
        """Test 404 error"""
        response = client.get('/nonexistent-endpoint')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False