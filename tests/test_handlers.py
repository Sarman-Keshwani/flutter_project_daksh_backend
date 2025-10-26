"""
Unit tests for API handlers.
"""
import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ["GEMINI_API_KEY"] = "test_api_key_12345"
os.environ["RATE_LIMIT_PER_MIN"] = "100"

from api.main import app
from api.gemini_client import GeminiError, GeminiTimeoutError

client = TestClient(app)

class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root health check."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test detailed health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["gemini_configured"] is True

class TestAIReplyEndpoint:
    """Test AI reply endpoint."""
    
    @patch('api.handlers.get_gemini_client')
    @pytest.mark.asyncio
    async def test_successful_reply(self, mock_get_client):
        """Test successful AI reply."""
        # Mock Gemini client
        mock_client = Mock()
        mock_client.generate_response = AsyncMock(return_value={
            "reply": "Hello! How can I help you?",
            "model": "gemini-1.5-flash",
            "prompt_tokens": 10,
            "completion_tokens": 20
        })
        mock_get_client.return_value = mock_client
        
        # Make request
        response = client.post(
            "/api/v1/ai/reply",
            json={"prompt": "Hello", "max_tokens": 100}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["reply"] == "Hello! How can I help you?"
        assert data["model"] == "gemini-1.5-flash"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 20
    
    def test_missing_prompt(self):
        """Test request with missing prompt."""
        response = client.post(
            "/api/v1/ai/reply",
            json={"max_tokens": 100}
        )
        assert response.status_code == 422  # Validation error
    
    def test_empty_prompt(self):
        """Test request with empty prompt."""
        response = client.post(
            "/api/v1/ai/reply",
            json={"prompt": "", "max_tokens": 100}
        )
        assert response.status_code == 422  # Validation error
    
    def test_invalid_max_tokens(self):
        """Test request with invalid max_tokens."""
        response = client.post(
            "/api/v1/ai/reply",
            json={"prompt": "Hello", "max_tokens": -1}
        )
        assert response.status_code == 422  # Validation error
    
    @patch('api.handlers.get_gemini_client')
    @pytest.mark.asyncio
    async def test_gemini_timeout(self, mock_get_client):
        """Test Gemini API timeout handling."""
        mock_client = Mock()
        mock_client.generate_response = AsyncMock(
            side_effect=GeminiTimeoutError("Timeout")
        )
        mock_get_client.return_value = mock_client
        
        response = client.post(
            "/api/v1/ai/reply",
            json={"prompt": "Hello", "max_tokens": 100}
        )
        
        assert response.status_code == 504  # Gateway timeout
    
    @patch('api.handlers.get_gemini_client')
    @pytest.mark.asyncio
    async def test_gemini_api_error(self, mock_get_client):
        """Test Gemini API error handling."""
        mock_client = Mock()
        mock_error = GeminiError("API Error", status_code=500)
        mock_client.generate_response = AsyncMock(side_effect=mock_error)
        mock_get_client.return_value = mock_client
        
        response = client.post(
            "/api/v1/ai/reply",
            json={"prompt": "Hello", "max_tokens": 100}
        )
        
        assert response.status_code == 502  # Bad gateway
    
    @patch('api.handlers.get_gemini_client')
    @pytest.mark.asyncio
    async def test_unexpected_error(self, mock_get_client):
        """Test unexpected error handling."""
        mock_client = Mock()
        mock_client.generate_response = AsyncMock(
            side_effect=Exception("Unexpected error")
        )
        mock_get_client.return_value = mock_client
        
        response = client.post(
            "/api/v1/ai/reply",
            json={"prompt": "Hello", "max_tokens": 100}
        )
        
        assert response.status_code == 500  # Internal server error
    
    def test_prompt_too_long(self):
        """Test request with excessively long prompt."""
        long_prompt = "a" * 10001  # Exceeds 10000 char limit
        response = client.post(
            "/api/v1/ai/reply",
            json={"prompt": long_prompt, "max_tokens": 100}
        )
        assert response.status_code == 422  # Validation error

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @patch('api.handlers.get_gemini_client')
    def test_rate_limit_not_exceeded_under_limit(self, mock_get_client):
        """Test that requests under the rate limit succeed."""
        # Mock successful responses
        mock_client = Mock()
        mock_client.generate_response = AsyncMock(return_value={
            "reply": "Test reply",
            "model": "gemini-1.5-flash",
            "prompt_tokens": 5,
            "completion_tokens": 10
        })
        mock_get_client.return_value = mock_client
        
        # Make multiple requests (under limit)
        for _ in range(5):
            response = client.post(
                "/api/v1/ai/reply",
                json={"prompt": "Test", "max_tokens": 100}
            )
            assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
