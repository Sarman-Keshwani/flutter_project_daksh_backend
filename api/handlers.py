"""
API handlers for AI endpoints.
"""
import os
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from .gemini_client import GeminiClient, GeminiError, GeminiTimeoutError

router = APIRouter(tags=["ai"])
limiter = Limiter(key_func=get_remote_address)

# Request/Response models
class AIRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt")
    max_tokens: Optional[int] = Field(512, ge=1, le=4096, description="Maximum tokens in response")

class UsageInfo(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

class AIResponse(BaseModel):
    id: str = Field(..., description="Unique response ID")
    reply: str = Field(..., description="AI generated reply")
    model: str = Field(..., description="Model identifier")
    usage: UsageInfo = Field(default_factory=UsageInfo, description="Token usage information")

# Gemini client instance
gemini_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    """Dependency to get or create Gemini client."""
    global gemini_client
    if gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY not configured"
            )
        gemini_client = GeminiClient(api_key=api_key)
    return gemini_client

@router.post("/ai/reply", response_model=AIResponse)
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MIN', '60')}/minute")
async def ai_reply(
    request: Request,
    ai_request: AIRequest,
    client: GeminiClient = Depends(get_gemini_client)
):
    """
    Process AI prompt and return response from Gemini.
    
    - **prompt**: User's input text (required)
    - **max_tokens**: Maximum tokens for response (optional, default: 512)
    
    Returns AI-generated reply with usage statistics.
    """
    try:
        # Generate unique response ID
        response_id = str(uuid.uuid4())
        
        # Call Gemini API
        result = await client.generate_response(
            prompt=ai_request.prompt,
            max_tokens=ai_request.max_tokens
        )
        
        return AIResponse(
            id=response_id,
            reply=result["reply"],
            model=result["model"],
            usage=UsageInfo(
                prompt_tokens=result.get("prompt_tokens"),
                completion_tokens=result.get("completion_tokens")
            )
        )
        
    except GeminiTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail="Request to Gemini AI timed out"
        )
    except GeminiError as e:
        # Map Gemini errors to appropriate HTTP codes
        status_code = 502  # Bad Gateway for upstream errors
        if hasattr(e, 'status_code'):
            if e.status_code == 401:
                status_code = 500  # Don't expose auth issues to client
            elif e.status_code == 429:
                status_code = 503  # Service unavailable
            else:
                status_code = 502
        
        raise HTTPException(
            status_code=status_code,
            detail=f"Gemini API error: {str(e)}"
        )
    except Exception as e:
        # Log error without exposing internals
        print(f"Unexpected error in ai_reply: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
