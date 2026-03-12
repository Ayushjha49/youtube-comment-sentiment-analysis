"""
=============================================================================
app.py — FastAPI backend for YouTube Sentiment Analysis
=============================================================================

ROUTES:
  POST /api/analyze  — Fetch comments + predict sentiment for a YouTube video
  GET  /api/health   — Health check / model status
  GET  /api/demo     — Demo response (no API key needed)

USAGE:
  cd backend
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

PRODUCTION:
  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse, HealthResponse
from predictor import SentimentPredictor, get_predictor

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(name)s — %(message)s',
)
logger = logging.getLogger(__name__)

# ── Load environment ──────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')
MAX_COMMENTS    = int(os.getenv('MAX_COMMENTS', '10000'))
DEMO_MODE       = os.getenv('DEMO_MODE', 'false').lower() == 'true'
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')


# ── App lifecycle ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models when server starts."""
    logger.info('🚀 Server starting — loading ML/DL models...')
    try:
        predictor = get_predictor()
        loaded    = []
        if predictor._ml_loaded:
            loaded.append('ml_ensemble')
        if predictor._dl_loaded:
            loaded.append('dl_bilstm')
        logger.info(f'✅ Models ready: {loaded}')
    except Exception as e:
        logger.warning(f'⚠️  Models not loaded at startup: {e}')
        logger.warning('   Train models first with training/train_ml.py and training/train_dl.py')

    yield
    logger.info('Server shutting down.')


# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title       = 'YouTube Sentiment Analysis API',
    description = 'Analyze sentiment of YouTube video comments using ML + DL',
    version     = '1.0.0',
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ['*'],
    allow_headers     = ['*'],
)


# ── Routes ────────────────────────────────────────────────────────────────
@app.get('/api/health', response_model=HealthResponse)
async def health_check():
    """Check API and model status."""
    predictor   = get_predictor()
    loaded      = []
    if predictor._ml_loaded:
        loaded.append('ml_ensemble')
    if predictor._dl_loaded:
        loaded.append('dl_bilstm')

    return HealthResponse(
        status        = 'ok' if loaded else 'degraded',
        models_loaded = loaded,
    )


@app.post('/api/analyze', response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    """
    Main endpoint: fetch YouTube comments and predict sentiment.

    Body:
        {
            "url": "https://youtube.com/watch?v=...",
            "max_comments": 500,
            "model": "ensemble"
        }
    """
    from src.youtube_fetcher import extract_video_id, fetch_comments, fetch_comments_mock
    from src.config import MAX_COMMENTS as CFG_MAX

    # ── Extract video ID ──────────────────────────────────────────────────
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code = 400,
            detail      = f'Invalid YouTube URL: {request.url!r}',
        )

    max_n = min(request.max_comments, CFG_MAX)

    # ── Fetch comments ────────────────────────────────────────────────────
    if not YOUTUBE_API_KEY or DEMO_MODE:
        logger.info(f'[API] Demo mode — using mock comments for video: {video_id}')
        comments, metadata = fetch_comments_mock(n=max_n)
        metadata['video_id'] = video_id
    else:
        try:
            comments, metadata = fetch_comments(
                video_id    = video_id,
                api_key     = YOUTUBE_API_KEY,
                max_comments= max_n,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=f'YouTube API error: {e}')

    if not comments:
        raise HTTPException(
            status_code = 404,
            detail      = 'No comments found for this video.',
        )

    # ── Run sentiment analysis ────────────────────────────────────────────
    try:
        predictor = get_predictor()
        result    = predictor.analyze_video(
            comments = comments,
            metadata = metadata,
            mode     = request.model,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code = 503,
            detail      = f'Model error: {e}. Please ensure models are trained.',
        )
    except Exception as e:
        logger.exception(f'Unexpected error during analysis: {e}')
        raise HTTPException(status_code=500, detail='Internal server error.')

    # ── Build response ────────────────────────────────────────────────────
    from schemas import SentimentDistribution
    return AnalyzeResponse(
        success              = True,
        video_id             = result.video_id,
        video_title          = result.video_title,
        channel              = result.channel,
        thumbnail            = result.thumbnail,
        total_comments_video = result.total_comments_video,
        analyzed_count       = result.analyzed_count,
        overall_sentiment    = result.overall_sentiment,
        overall_confidence   = result.overall_confidence,
        distribution         = SentimentDistribution(
            positive = result.distribution.get('positive', 0),
            negative = result.distribution.get('negative', 0),
            neutral  = result.distribution.get('neutral', 0),
        ),
        top_positive         = result.top_positive,
        top_negative         = result.top_negative,
        processing_time_s    = result.processing_time_s,
        model_used           = result.model_used,
    )


@app.get('/api/demo')
async def demo():
    """
    Returns a demo analysis result without needing API key or trained models.
    Useful for testing the frontend.
    """
    return {
        'success'              : True,
        'video_id'             : 'dQw4w9WgXcQ',
        'video_title'          : 'Demo Video — Sentiment Analysis Example',
        'channel'              : 'Demo Channel',
        'thumbnail'            : '',
        'total_comments_video' : 12500,
        'analyzed_count'       : 500,
        'overall_sentiment'    : 'positive',
        'overall_confidence'   : 0.72,
        'distribution'         : {
            'positive' : 61.4,
            'negative' : 18.8,
            'neutral'  : 19.8,
        },
        'top_positive': [
            'This video is absolutely amazing! Best content ever 🔥',
            'ekdam ramro video thiyo yaar, helpful',
            'Great explanation, finally understood this properly',
        ],
        'top_negative': [
            'kasto bakwas video ho, time waste',
            'Worst video ever, completely misleading',
        ],
        'processing_time_s': 2.34,
        'model_used'       : 'ml+dl_ensemble',
    }


# ── Exception handlers ────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code = exc.status_code,
        content     = ErrorResponse(
            success = False,
            error   = str(exc.detail),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f'Unhandled exception: {exc}')
    return JSONResponse(
        status_code = 500,
        content     = ErrorResponse(
            success = False,
            error   = 'Internal server error',
            detail  = str(exc),
        ).dict(),
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'app:app',
        host    = '0.0.0.0',
        port    = 8000,
        reload  = True,
        log_level = 'info',
    )
