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
"""

import os
import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

YOUTUBE_API_BASE = 'https://www.googleapis.com/youtube/v3'


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

    # Already a bare video ID (11 chars, alphanumeric + - _)
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ''

        # youtu.be short URL
        if 'youtu.be' in hostname:
            vid = parsed.path.lstrip('/')
            return vid.split('?')[0] if vid else None

        # youtube.com/shorts/
        if 'shorts' in parsed.path:
            parts = parsed.path.split('/')
            idx = parts.index('shorts')
            if idx + 1 < len(parts):
                return parts[idx + 1]

        # youtube.com/watch?v=
        if 'youtube.com' in hostname or 'youtube' in hostname:
            qs = parse_qs(parsed.query)
            return qs.get('v', [None])[0]

    except Exception as e:
        logger.warning(f'Failed to parse URL {url}: {e}')

    return None


def get_video_metadata(video_id: str, api_key: str) -> Dict:
    """Fetch video title, channel, view count, like count, comment count."""
    url = f'{YOUTUBE_API_BASE}/videos'
    params = {
        'part': 'snippet,statistics',
        'id': video_id,
        'key': api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get('items'):
        raise ValueError(f'Video not found: {video_id}')

    item = data['items'][0]
    snippet = item.get('snippet', {})
    stats   = item.get('statistics', {})

    return {
        'video_id'       : video_id,
        'title'          : snippet.get('title', 'Unknown'),
        'channel'        : snippet.get('channelTitle', 'Unknown'),
        'published_at'   : snippet.get('publishedAt', ''),
        'view_count'     : int(stats.get('viewCount', 0)),
        'like_count'     : int(stats.get('likeCount', 0)),
        'comment_count'  : int(stats.get('commentCount', 0)),
        'thumbnail'      : snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
    }


def fetch_comments(
    video_id: str,
    api_key: str,
    max_comments: int = 500,
    order: str = 'relevance',
    include_replies: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    Fetch top-level comments from a YouTube video.

    Returns:
        comments   : List of dicts with 'text', 'likes', 'author'
        metadata   : Video metadata dict
    """
    if not api_key:
        raise ValueError(
            'YouTube API key not set. '
            'Add YOUTUBE_API_KEY to your .env file.'
        )

    metadata = get_video_metadata(video_id, api_key)
    logger.info(f'[YT] Fetching comments for: "{metadata["title"]}"')
    logger.info(f'[YT] Total comments on video: {metadata["comment_count"]:,}')

    comments = []
    page_token = None
    page_size = min(100, max_comments)  # YouTube max per page = 100

    while len(comments) < max_comments:
        params = {
            'part'       : 'snippet',
            'videoId'    : video_id,
            'key'        : api_key,
            'maxResults' : page_size,
            'order'      : order,
            'textFormat' : 'plainText',
        }
        if page_token:
            params['pageToken'] = page_token

        try:
            resp = requests.get(
                f'{YOUTUBE_API_BASE}/commentThreads',
                params=params,
                timeout=15,
            )

            if resp.status_code == 403:
                error_msg = resp.json().get('error', {}).get('message', '')
                if 'commentsDisabled' in error_msg or 'disabled' in error_msg.lower():
                    raise ValueError('Comments are disabled for this video.')
                raise ValueError(f'API quota exceeded or forbidden: {error_msg}')

            if resp.status_code == 404:
                raise ValueError(f'Video not found: {video_id}')

            resp.raise_for_status()
            data = resp.json()

        except requests.exceptions.Timeout:
            logger.warning('[YT] Request timed out, retrying...')
            time.sleep(2)
            continue
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f'YouTube API request failed: {e}')

        for item in data.get('items', []):
            top_comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'text'       : top_comment.get('textDisplay', ''),
                'likes'      : top_comment.get('likeCount', 0),
                'author'     : top_comment.get('authorDisplayName', 'Anonymous'),
                'published'  : top_comment.get('publishedAt', ''),
                'reply_count': item['snippet'].get('totalReplyCount', 0),
            })

        page_token = data.get('nextPageToken')
        if not page_token:
            break

        # Small delay to be a good API citizen
        time.sleep(0.1)

    logger.info(f'[YT] Fetched {len(comments)} comments')
    return comments[:max_comments], metadata


def comments_to_texts(comments: List[Dict]) -> List[str]:
    """Extract just the text from comment dicts."""
    return [c['text'] for c in comments if c.get('text', '').strip()]


# ── Mock fetcher for testing without API key ───────────────────────────────
MOCK_COMMENTS = [
    {"text": "This video is absolutely amazing! Best content on YouTube 🔥", "likes": 245},
    {"text": "vayo ni yaar ekdam ramro video thiyo, dherai helpful", "likes": 89},
    {"text": "kasto bakwas video ho, time waste bhayo mero", "likes": 12},
    {"text": "Great explanation, finally understood this topic properly", "likes": 156},
    {"text": "ekdam bekar content, subscribe nai garnuparena", "likes": 3},
    {"text": "Bro your videos are always so helpful, keep it up! 👍", "likes": 78},
    {"text": "okay video ho, not bad not great", "likes": 45},
    {"text": "ramro thiyo tara aru ramro garna sakincha ni", "likes": 34},
    {"text": "Worst video I've ever watched, completely misleading", "likes": 67},
    {"text": "Thank you so much for this! It really helped me 🙏", "likes": 203},
]

MOCK_METADATA = {
    "video_id": "demo_video",
    "title": "Demo Video — Mock Mode",
    "channel": "Test Channel",
    "view_count": 50000,
    "like_count": 2000,
    "comment_count": 850,
    "thumbnail": "",
    "published_at": "2024-01-01T00:00:00Z",
}


def fetch_comments_mock(n: int = 100) -> Tuple[List[Dict], Dict]:
    """Return mock comments for testing without API key."""
    import random
    comments = (MOCK_COMMENTS * (n // len(MOCK_COMMENTS) + 1))[:n]
    # Add some randomness
    for c in comments:
        c = dict(c)
    return comments, MOCK_METADATA


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    test_urls = [
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'https://youtu.be/dQw4w9WgXcQ',
        'dQw4w9WgXcQ',
    ]
    for url in test_urls:
        vid_id = extract_video_id(url)
        print(f'{url!r} → {vid_id!r}')
