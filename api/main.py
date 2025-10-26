"""
FastAPI main application entry point for Vercel deployment.
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

from .handlers import router

# Load environment variables
load_dotenv()

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# FastAPI app
app = FastAPI(
    title="Gemini AI Backend",
    description="FastAPI backend for proxying requests to Gemini AI",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware for Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production to your Flutter web domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Gemini AI Backend",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    return {
        "status": "ok",
        "gemini_configured": bool(gemini_api_key),
        "rate_limit": os.getenv("RATE_LIMIT_PER_MIN", "60")
    }

# Vercel serverless handler
handler = app
