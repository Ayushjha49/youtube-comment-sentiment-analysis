"""
=============================================================================
youtube_fetcher.py — Fetch comments from YouTube videos via Data API v3
=============================================================================

SETUP:
  1. Go to https://console.cloud.google.com
  2. Enable "YouTube Data API v3"
  3. Create API credentials → API Key
  4. Set YOUTUBE_API_KEY in your .env file

API QUOTA NOTE:
  YouTube API has a quota of 10,000 units/day (free).
  Each commentThreads.list call costs ~1 unit.
  With pagination (100 comments/page), 500 comments = ~5 API calls.

PERFORMANCE NOTES:
  - requests.Session() reuses the underlying TCP connection across all pages,
    avoiding the handshake overhead of opening a new connection every request.
    For 190 pages (19k comments) this saves meaningful latency.

  - REQUEST_DELAY_S defaults to 0.0 (no sleep between pages). The original
    0.1s sleep added ~19s of pure waiting on a 190-page fetch. Only set this
    above 0 in .env (YT_REQUEST_DELAY_S=0.1) if you hit 429 rate-limit errors.

  - fetch_comments_pipelined() starts a background thread immediately and
    returns (queue, metadata). The caller begins predicting page 1 while
    page 2 is still being fetched, overlapping network I/O with model compute.
"""

import os
import re
import time
import queue
import logging
import threading
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests
from dotenv import load_dotenv
from src.utils import timing_decorator

load_dotenv()
logger = logging.getLogger(__name__)

YOUTUBE_API_BASE = 'https://www.googleapis.com/youtube/v3'

# Delay between page requests. Default 0.0 — no delay.
# The original 0.1s sleep wasted ~19s per 190-page fetch for no benefit.
# Only increase if you are hitting 429 rate-limit errors from the API.
# Configure via .env: YT_REQUEST_DELAY_S=0.1
try:
    REQUEST_DELAY_S = max(0.0, float(os.getenv('YT_REQUEST_DELAY_S', '0.0')))
except ValueError:
    REQUEST_DELAY_S = 0.0


# =============================================================================
# URL PARSING
# =============================================================================
def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats:
      - https://www.youtube.com/watch?v=dQw4w9WgXcQ
      - https://youtu.be/dQw4w9WgXcQ
      - https://youtube.com/shorts/dQw4w9WgXcQ
      - https://m.youtube.com/watch?v=dQw4w9WgXcQ
      - dQw4w9WgXcQ (bare ID)
    """
    url = url.strip()

    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url

    try:
        parsed   = urlparse(url)
        hostname = parsed.hostname or ''

        if 'youtu.be' in hostname:
            vid = parsed.path.lstrip('/')
            return vid.split('?')[0] if vid else None

        if 'shorts' in parsed.path:
            parts = parsed.path.split('/')
            idx   = parts.index('shorts')
            if idx + 1 < len(parts):
                return parts[idx + 1]

        if 'youtube.com' in hostname or 'youtube' in hostname:
            qs = parse_qs(parsed.query)
            return qs.get('v', [None])[0]

    except Exception as e:
        logger.warning(f'Failed to parse URL {url}: {e}')

    return None


# =============================================================================
# VIDEO METADATA
# =============================================================================
@timing_decorator
def get_video_metadata(video_id: str, api_key: str) -> Dict:
    """Fetch video title, channel, view count, like count, comment count."""
    url    = f'{YOUTUBE_API_BASE}/videos'
    params = {
        'part': 'snippet,statistics',
        'id'  : video_id,
        'key' : api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get('items'):
        raise ValueError(f'Video not found: {video_id}')

    item    = data['items'][0]
    snippet = item.get('snippet', {})
    stats   = item.get('statistics', {})

    return {
        'video_id'     : video_id,
        'title'        : snippet.get('title', 'Unknown'),
        'channel'      : snippet.get('channelTitle', 'Unknown'),
        'published_at' : snippet.get('publishedAt', ''),
        'view_count'   : int(stats.get('viewCount', 0)),
        'like_count'   : int(stats.get('likeCount', 0)),
        'comment_count': int(stats.get('commentCount', 0)),
        'thumbnail'    : snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
    }


def _parse_comment_item(item: Dict) -> Dict:
    """Extract comment fields from a YouTube API commentThread item."""
    top = item['snippet']['topLevelComment']['snippet']
    return {
        'text'       : top.get('textDisplay', ''),
        'likes'      : top.get('likeCount', 0),
        'author'     : top.get('authorDisplayName', 'Anonymous'),
        'published'  : top.get('publishedAt', ''),
        'reply_count': item['snippet'].get('totalReplyCount', 0),
    }


def _handle_error_response(resp: requests.Response, video_id: str) -> None:
    """Raise appropriate exception for non-2xx API responses."""
    if resp.status_code == 403:
        msg = resp.json().get('error', {}).get('message', '')
        if 'commentsDisabled' in msg or 'disabled' in msg.lower():
            raise ValueError('Comments are disabled for this video.')
        raise ValueError(f'API quota exceeded or forbidden: {msg}')
    if resp.status_code == 404:
        raise ValueError(f'Video not found: {video_id}')
    resp.raise_for_status()


# =============================================================================
# SEQUENTIAL FETCH — used as fallback / for demo mode
# =============================================================================
@timing_decorator
def fetch_comments(
    video_id      : str,
    api_key       : str,
    max_comments  : int = 500,
    order         : str = 'time',
    include_replies: bool = False,
    fetch_timeout : int = 0,
) -> Tuple[List[Dict], Dict]:
    """
    Fetch top-level comments from a YouTube video (sequential, blocking).

    Uses requests.Session() to reuse the TCP connection across all pages.

    Args:
        fetch_timeout : Stop fetching after this many seconds and return
                        whatever has been collected so far. 0 = no limit.
                        Configure via .env: FETCH_TIMEOUT_SECONDS=120
    Returns:
        comments : List of dicts with 'text', 'likes', 'author', etc.
        metadata : Video metadata dict
    """
    if not api_key:
        raise ValueError(
            'YouTube API key not set. '
            'Add YOUTUBE_API_KEY to your .env file.'
        )

    metadata = get_video_metadata(video_id, api_key)
    logger.info(f'[YT] Fetching comments for: "{metadata["title"]}"')
    logger.info(f'[YT] Total comments on video: {metadata["comment_count"]:,}')

    if fetch_timeout > 0:
        logger.info(f'[YT] Fetch time limit: {fetch_timeout}s')

    comments    = []
    page_token  = None
    fetch_start = time.time()

    with requests.Session() as session:
        while len(comments) < max_comments:

            # Time-limit check
            if fetch_timeout > 0 and (time.time() - fetch_start) >= fetch_timeout:
                logger.info(
                    f'[YT] Fetch timeout reached ({fetch_timeout}s). '
                    f'Collected {len(comments)} comments.'
                )
                break

            params = {
                'part'      : 'snippet',
                'videoId'   : video_id,
                'key'       : api_key,
                'maxResults': min(100, max_comments - len(comments)),
                'order'     : order,
                'textFormat': 'plainText',
            }
            if page_token:
                params['pageToken'] = page_token

            try:
                resp = session.get(
                    f'{YOUTUBE_API_BASE}/commentThreads',
                    params  = params,
                    timeout = 15,
                )
                _handle_error_response(resp, video_id)
                data = resp.json()

            except (ValueError, RuntimeError):
                raise
            except requests.exceptions.Timeout:
                logger.warning('[YT] Request timed out, retrying...')
                time.sleep(2)
                continue
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f'YouTube API request failed: {e}')

            for item in data.get('items', []):
                comments.append(_parse_comment_item(item))

            page_token = data.get('nextPageToken')
            if not page_token:
                break

            if REQUEST_DELAY_S > 0:
                time.sleep(REQUEST_DELAY_S)

    fetch_elapsed = time.time() - fetch_start
    logger.info(f'[YT] Fetched {len(comments)} comments in {fetch_elapsed:.1f}s')
    return comments[:max_comments], metadata


# =============================================================================
# PIPELINED FETCH — fetches in background thread, yields pages via queue
# =============================================================================
@timing_decorator
def fetch_comments_pipelined(
    video_id     : str,
    api_key      : str,
    max_comments : int = 500,
    order        : str = 'time',
    fetch_timeout: int = 0,
    queue_maxsize: int = 5,
) -> Tuple[queue.Queue, Dict]:
    """
    Start fetching comments in a background thread. Returns immediately.

    The caller reads pages from page_queue:
      - Each item is a List[Dict] (one page = up to 100 comments)
      - Sentinel None signals fetch complete
      - An Exception object signals a fetch error

    This allows the caller to begin predicting page 1 while page 2 is still
    being fetched — overlapping network I/O with model compute.

    Timeline:
      Fetch thread: [p1][p2][p3]...[pN][None]
      Main thread:      [pred1][pred2]...[predN][aggregate]
      Total ≈ max(fetch_time, predict_time) vs fetch_time + predict_time

    Args:
        queue_maxsize : Max pages buffered at once (backpressure).
                        If predictor is slow, fetcher blocks here rather
                        than pulling all 19k comments into memory at once.
    Returns:
        page_queue : Queue to read comment pages from
        metadata   : Video metadata (fetched synchronously before thread starts)
    """
    if not api_key:
        raise ValueError(
            'YouTube API key not set. '
            'Add YOUTUBE_API_KEY to your .env file.'
        )

    # Fetch metadata synchronously — needed for the response object
    metadata = get_video_metadata(video_id, api_key)
    logger.info(
        f'[YT-Pipeline] Fetching: "{metadata["title"]}" | '
        f'{metadata["comment_count"]:,} total comments'
    )

    page_queue = queue.Queue(maxsize=queue_maxsize)

    def _fetch_worker():
        """Background thread: fetch pages and push to queue."""
        fetched    = 0
        page_token = None
        start_time = time.time()

        try:
            with requests.Session() as session:
                while fetched < max_comments:

                    # Time-limit check
                    if fetch_timeout > 0 and (time.time() - start_time) >= fetch_timeout:
                        logger.info(
                            f'[YT-Pipeline] Timeout {fetch_timeout}s reached. '
                            f'Fetched {fetched} comments total.'
                        )
                        break

                    params = {
                        'part'      : 'snippet',
                        'videoId'   : video_id,
                        'key'       : api_key,
                        'maxResults': min(100, max_comments - fetched),
                        'order'     : order,
                        'textFormat': 'plainText',
                    }
                    if page_token:
                        params['pageToken'] = page_token

                    try:
                        resp = session.get(
                            f'{YOUTUBE_API_BASE}/commentThreads',
                            params  = params,
                            timeout = 15,
                        )
                        _handle_error_response(resp, video_id)
                        data = resp.json()

                    except (ValueError, RuntimeError) as e:
                        page_queue.put(e)
                        return
                    except requests.exceptions.Timeout:
                        logger.warning('[YT-Pipeline] Page timed out, retrying...')
                        time.sleep(2)
                        continue
                    except requests.exceptions.RequestException as e:
                        page_queue.put(RuntimeError(f'YouTube API request failed: {e}'))
                        return

                    page_batch = [
                        _parse_comment_item(item)
                        for item in data.get('items', [])
                    ]

                    if page_batch:
                        fetched += len(page_batch)
                        # Blocks if queue is full (backpressure)
                        page_queue.put(page_batch)
                        logger.debug(
                            f'[YT-Pipeline] Page queued: {len(page_batch)} comments '
                            f'(total: {fetched})'
                        )

                    page_token = data.get('nextPageToken')
                    if not page_token:
                        break

                    if REQUEST_DELAY_S > 0:
                        time.sleep(REQUEST_DELAY_S)

        except Exception as e:
            page_queue.put(e)
            return
        finally:
            # Always signal completion so consumer never hangs
            page_queue.put(None)
            elapsed = time.time() - start_time
            logger.info(
                f'[YT-Pipeline] Fetch thread done: {fetched} comments in {elapsed:.1f}s'
            )

    thread = threading.Thread(target=_fetch_worker, daemon=True)
    thread.start()

    return page_queue, metadata


# =============================================================================
# UTILITIES
# =============================================================================
def comments_to_texts(comments: List[Dict]) -> List[str]:
    """Extract just the text from comment dicts."""
    return [c['text'] for c in comments if c.get('text', '').strip()]


# ── Mock fetcher for testing without API key ──────────────────────────────
MOCK_COMMENTS = [
    {"text": "This video is absolutely amazing! Best content on YouTube 🔥", "likes": 245},
    {"text": "vayo ni yaar ekdam ramro video thiyo, dherai helpful",           "likes": 89},
    {"text": "kasto bakwas video ho, time waste bhayo mero",                   "likes": 12},
    {"text": "Great explanation, finally understood this topic properly",       "likes": 156},
    {"text": "ekdam bekar content, subscribe nai garnuparena",                 "likes": 3},
    {"text": "Bro your videos are always so helpful, keep it up! 👍",          "likes": 78},
    {"text": "okay video ho, not bad not great",                               "likes": 45},
    {"text": "ramro thiyo tara aru ramro garna sakincha ni",                   "likes": 34},
    {"text": "Worst video I've ever watched, completely misleading",            "likes": 67},
    {"text": "Thank you so much for this! It really helped me 🙏",             "likes": 203},
]

MOCK_METADATA = {
    "video_id"     : "demo_video",
    "title"        : "Demo Video — Mock Mode",
    "channel"      : "Test Channel",
    "view_count"   : 50000,
    "like_count"   : 2000,
    "comment_count": 850,
    "thumbnail"    : "",
    "published_at" : "2024-01-01T00:00:00Z",
}


def fetch_comments_mock(n: int = 100) -> Tuple[List[Dict], Dict]:
    """Return mock comments for testing without API key."""
    comments = (MOCK_COMMENTS * (n // len(MOCK_COMMENTS) + 1))[:n]
    return comments, MOCK_METADATA


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_urls = [
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'https://youtu.be/dQw4w9WgXcQ',
        'dQw4w9WgXcQ',
    ]
    for url in test_urls:
        vid_id = extract_video_id(url)
        print(f'{url!r} → {vid_id!r}')
