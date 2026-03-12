"""
=============================================================================
schemas.py — Pydantic request/response models for FastAPI
=============================================================================
"""

from pydantic import BaseModel, HttpUrl, validator
from typing import Dict, List, Optional


# ── Request ────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    url        : str
    max_comments: int = 1000
    model      : str  = 'ensemble'   # 'ml', 'dl', 'ensemble'

    @validator('max_comments')
    def clamp_comments(cls, v):
        return max(10, min(v, 20000))

    @validator('model')
    def validate_model(cls, v):
        if v not in ('ml', 'dl', 'ensemble'):
            raise ValueError("model must be 'ml', 'dl', or 'ensemble'")
        return v


# ── Response ───────────────────────────────────────────────────────────────
class SentimentDistribution(BaseModel):
    positive: float
    negative: float
    neutral : float


class CommentSample(BaseModel):
    text      : str
    sentiment : str
    confidence: float


class AnalyzeResponse(BaseModel):
    success             : bool
    video_id            : str
    video_title         : str
    channel             : str
    thumbnail           : str
    total_comments_video: int
    analyzed_count      : int
    overall_sentiment   : str
    overall_confidence  : float
    distribution        : SentimentDistribution
    top_positive        : List[str]
    top_negative        : List[str]
    processing_time_s   : float
    model_used          : str


class ErrorResponse(BaseModel):
    success: bool = False
    error  : str
    detail : Optional[str] = None


class HealthResponse(BaseModel):
    status       : str
    models_loaded: List[str]
    version      : str = '1.0.0'
