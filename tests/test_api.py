#!/usr/bin/env python3
"""Basic API tests for TTS backend service"""
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from fastapi.testclient import TestClient
from config import get_default_config
from api.endpoints import create_app


def test_imports():
    """Test that imports work correctly"""
    assert "api" in sys.modules
    assert "models" in sys.modules
    assert "utils" in sys.modules
    print("✓ All imports successful")


def test_app_creation():
    """Test FastAPI app creation"""
    config = get_default_config()
    app = create_app(config)
    
    assert app is not None
    assert hasattr(app, "openapi")
    print("✓ FastAPI app created successfully")


def test_api_endpoints():
    """Test basic API endpoints"""
    config = get_default_config()
    app = create_app(config)
    
    # Test root endpoint - direct call
    from fastapi import Request
    from fastapi.responses import JSONResponse
    
    # Simulate request by using app directly
    # We'll use app.router to get the route handlers
    try:
        # Try to use TestClient with new API
        client = TestClient(app, base_url="http://testserver")
    except TypeError:
        # Fallback to old API
        client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "TTS Backend"
    print("✓ Root endpoint works")
    
    # Test health check
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    print("✓ Health check endpoint works")
    
    # Test get folders
    response = client.get("/get_folders")
    assert response.status_code == 200
    data = response.json()
    assert "samples_dir" in data
    assert "output_dir" in data
    print("✓ Folders endpoint works")
    
    # Test languages endpoint
    response = client.get("/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    print("✓ Languages endpoint works")
    
    # Test models list endpoint
    response = client.get("/get_models_list")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    print("✓ Models list endpoint works")
    
    # Test speakers list endpoint (should work even with empty samples dir)
    response = client.get("/speakers_list")
    assert response.status_code == 200
    data = response.json()
    assert "speakers" in data
    assert "count" in data
    print("✓ Speakers list endpoint works")


def test_invalid_endpoints():
    """Test error handling for invalid endpoints"""
    config = get_default_config()
    app = create_app(config)
    client = TestClient(app)
    
    # Test non-existent endpoint
    response = client.get("/nonexistent")
    assert response.status_code == 404
    print("✓ Invalid endpoints return 404")


def test_tts_stream_unsupported():
    """Test that streaming endpoint returns appropriate error"""
    config = get_default_config()
    app = create_app(config)
    client = TestClient(app)
    
    # Test streaming endpoint (should return 501 Not Implemented)
    response = client.get("/tts_stream?text=test&speaker_wav=test.wav&language=en")
    assert response.status_code == 501
    print("✓ Streaming endpoint correctly returns unsupported")


def test_schema_validation():
    """Test Pydantic schema validation"""
    from api.schemas import SynthesisRequest, SpeakerInfo, LanguageInfo
    
    # Test SynthesisRequest validation
    request = SynthesisRequest(
        text="Hello world",
        speaker_wav="test.wav",
        language="en"
    )
    assert request.text == "Hello world"
    assert request.language == "en"
    
    # Test SpeakerInfo validation (with optional fields)
    speaker = SpeakerInfo(
        name="test_speaker",
        type="audio_only",
        file_path="test.wav",
        text_path=None,
        design_path=None,
        design_description=None,
        voice_id=None  # Will be set by model_validator
    )
    assert speaker.name == "test_speaker"
    assert speaker.type == "audio_only"
    assert speaker.file_path == "test.wav"
    assert speaker.text_path is None
    assert speaker.design_path is None
    assert speaker.design_description is None
    assert speaker.voice_id == "test_speaker"  # Should be set to name
    
    # Test LanguageInfo validation
    language = LanguageInfo(
        code="en",
        name="English"
    )
    assert language.code == "en"
    assert language.name == "English"
    
    print("✓ Schema validation works correctly")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running TTS Backend API Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_app_creation,
        test_schema_validation,
        test_api_endpoints,
        test_invalid_endpoints,
        test_tts_stream_unsupported,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed: {e}")
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    # Run tests
    sys.exit(main())