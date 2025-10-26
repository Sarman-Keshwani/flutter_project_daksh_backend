"""
Gemini API client for making requests to Google's Gemini AI service.
"""
import os
import httpx
from typing import Dict, Any, Optional

# Gemini API configuration
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta"
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))

class GeminiError(Exception):
    """Base exception for Gemini API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class GeminiTimeoutError(GeminiError):
    """Exception for Gemini API timeout."""
    pass

class GeminiClient:
    """Client for interacting with Gemini AI API."""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key for authentication
        """
        self.api_key = api_key
        self.base_url = GEMINI_BASE_URL
        self.model = GEMINI_MODEL
        self.timeout = GEMINI_TIMEOUT
        
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = 512
    ) -> Dict[str, Any]:
        """
        Generate response from Gemini AI.
        
        Args:
            prompt: User's input prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing:
                - reply: AI generated text
                - model: Model identifier
                - prompt_tokens: Number of prompt tokens (optional)
                - completion_tokens: Number of completion tokens (optional)
                
        Raises:
            GeminiTimeoutError: If request times out
            GeminiError: For other API errors
        """
        url = f"{self.base_url}/models/{self.model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Build request payload for Gemini API
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 40
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    params={"key": self.api_key}
                )
                
                # Handle HTTP errors
                if response.status_code != 200:
                    error_detail = "Unknown error"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get("error", {}).get("message", error_detail)
                    except:
                        error_detail = response.text[:200]
                    
                    raise GeminiError(
                        f"Gemini API error: {error_detail}",
                        status_code=response.status_code
                    )
                
                data = response.json()
                
                # Extract response text
                reply = self._extract_reply(data)
                
                # Extract usage information if available
                usage_metadata = data.get("usageMetadata", {})
                prompt_tokens = usage_metadata.get("promptTokenCount")
                completion_tokens = usage_metadata.get("candidatesTokenCount")
                
                return {
                    "reply": reply,
                    "model": self.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
                
        except httpx.TimeoutException:
            raise GeminiTimeoutError("Request to Gemini API timed out")
        except GeminiError:
            raise
        except Exception as e:
            raise GeminiError(f"Unexpected error: {str(e)}")
    
    def _extract_reply(self, data: Dict[str, Any]) -> str:
        """
        Extract reply text from Gemini API response.
        
        Args:
            data: API response data
            
        Returns:
            Extracted reply text
            
        Raises:
            GeminiError: If response format is invalid
        """
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise GeminiError("No candidates in response")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise GeminiError("No parts in response content")
            
            reply = parts[0].get("text", "")
            
            if not reply:
                raise GeminiError("Empty reply from Gemini")
            
            return reply
            
        except (KeyError, IndexError, TypeError) as e:
            raise GeminiError(f"Invalid response format: {str(e)}")
