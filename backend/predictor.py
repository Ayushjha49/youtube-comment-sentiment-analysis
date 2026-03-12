"""
=============================================================================
predictor.py — Unified inference engine for ML + DL sentiment prediction
=============================================================================

PREDICTION FLOW:
  1. Raw comment texts  →  TextCleaner  →  Cleaned texts (once, shared)
  2a. Cleaned texts  →  TF-IDF  →  ML Ensemble  →  probabilities  ┐ parallel
  2b. Cleaned texts  →  Tokenize/Pad  →  BiLSTM  →  probabilities  ┘
  3. Weighted blend of ML + DL probabilities  →  Final predictions
  4. Aggregate per-comment predictions  →  Video-level summary

OPTIMISATIONS (compared to naive implementation):
  - Clean texts ONCE, pass shared result to both ML and DL (avoids double work)
  - ML and DL run in PARALLEL via ThreadPoolExecutor (sklearn + TF release GIL)
  - BiLSTM batch_size=256 (vs 64): ~4x fewer forward passes, ~2x faster inference
  - Parallel batch_clean across 4 threads for large batches
  - Skip per-comment detail objects for large batches (>MAX_DETAILED_COMMENTS)
    — they are never used by the API response, only top-5 comments are returned
  - No redundant TextCleaner instantiation inside analyze_video loops

CACHING:
  Models are loaded once at startup and kept in memory. Thread-safe because
  models are read-only after load (no weight updates at inference time).
"""

import os
import sys
import pickle
import logging
import time
import numpy as np
import queue as Q
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import timing_decorator

logger = logging.getLogger(__name__)


@dataclass
class CommentPrediction:
    text      : str
    sentiment : str             # 'positive', 'negative', 'neutral'
    confidence: float
    scores    : Dict[str, float]  # {'positive': 0.7, 'negative': 0.1, 'neutral': 0.2}


@dataclass
class VideoSentimentResult:
    video_id            : str
    video_title         : str
    channel             : str
    thumbnail           : str
    total_comments_video: int
    analyzed_count      : int
    overall_sentiment   : str
    overall_confidence  : float
    distribution        : Dict[str, float]   # percentages
    avg_scores          : Dict[str, float]   # average probabilities
    top_positive        : List[str]
    top_negative        : List[str]
    processing_time_s   : float
    model_used          : str
    comment_predictions : List[CommentPrediction] = field(default_factory=list)


class SentimentPredictor:
    """
    Unified predictor that loads and runs ML and/or DL models.

    Lazy-loads models on first use and caches them in memory.
    Thread-safe for FastAPI concurrent requests (models are read-only after load).
    """

    LABELS = ['negative', 'neutral', 'positive']

    # For batches larger than this, skip building per-comment CommentPrediction
    # objects. The API response only uses top-5 comments anyway, not the full
    # per-comment payload. Skips ~5-8s of unnecessary object creation for 19k comments.
    # Configure via .env: MAX_DETAILED_COMMENTS=2000
    MAX_DETAILED_COMMENTS = int(os.getenv('MAX_DETAILED_COMMENTS', '2000'))

    def __init__(self, config=None):
        if config is None:
            from src.config import InferenceConfig, ModelFiles
            self.cfg   = InferenceConfig
            self.files = ModelFiles
        else:
            self.cfg   = config['inference']
            self.files = config['files']

        self._ml_model     = None
        self._tfidf        = None
        self._ml_le        = None
        self._dl_model     = None
        self._dl_tokenizer = None
        self._dl_cleaner   = None
        self._dl_le        = None
        self._ml_cleaner   = None

        self._ml_loaded = False
        self._dl_loaded = False

    # ── Model Loading ─────────────────────────────────────────────────────
    @timing_decorator
    def load_ml(self):
        """Load ML ensemble + TF-IDF vectorizers."""
        if self._ml_loaded:
            return
        logger.info('[Predictor] Loading ML ensemble...')
        try:
            from src.preprocess import TFIDFExtractor, TextCleaner
            self._tfidf      = TFIDFExtractor.load(self.files.TFIDF_WORD, self.files.TFIDF_CHAR)
            with open(self.files.ML_ENSEMBLE, 'rb') as f:
                self._ml_model = pickle.load(f)
            with open(self.files.LABEL_ENCODER, 'rb') as f:
                self._ml_le = pickle.load(f)
            self._ml_cleaner = TextCleaner(remove_stopwords=True)
            self._ml_loaded  = True
            logger.info('[Predictor] ML models loaded ✓')
        except FileNotFoundError as e:
            logger.error(f'[Predictor] ML model file missing: {e}')
            raise RuntimeError(
                f'ML models not found. Run training/train_ml.py first. ({e})'
            )

    @timing_decorator
    def load_dl(self):
        """Load BiLSTM model + tokenizer."""
        if self._dl_loaded:
            return
        logger.info('[Predictor] Loading DL model...')
        try:
            from src.dl_model import load_dl_model, AttentionLayer
            self._dl_model = load_dl_model(self.files.DL_MODEL)
            with open(self.files.DL_TOKENIZER, 'rb') as f:
                self._dl_tokenizer = pickle.load(f)
            with open(self.files.DL_LABEL_ENCODER, 'rb') as f:
                self._dl_le = pickle.load(f)
            with open(self.files.DL_TEXT_CLEANER, 'rb') as f:
                self._dl_cleaner = pickle.load(f)
            self._dl_loaded  = True
            logger.info('[Predictor] DL model loaded ✓')
        except FileNotFoundError as e:
            logger.error(f'[Predictor] DL model file missing: {e}')
            raise RuntimeError(
                f'DL model not found. Run training/train_dl.py first. ({e})'
            )

    @timing_decorator
    def load_all(self):
        """Load all available models."""
        loaded = []
        try:
            self.load_ml()
            loaded.append('ml')
        except RuntimeError as e:
            logger.warning(f'[Predictor] Could not load ML: {e}')

        try:
            self.load_dl()
            loaded.append('dl')
        except RuntimeError as e:
            logger.warning(f'[Predictor] Could not load DL: {e}')

        if not loaded:
            raise RuntimeError('No models available. Please train models first.')

        return loaded

    # ── Individual Model Inference ─────────────────────────────────────────
    @timing_decorator
    def predict_ml(self, texts: List[str], cleaned: List[str] = None) -> np.ndarray:
        """
        Run ML ensemble inference. Returns (N, 3) probability array.

        Args:
            cleaned : Pre-cleaned texts. Pass this to skip cleaning when ML
                      and DL share the same cleaned batch (avoids double work).
        """
        self.load_ml()
        if cleaned is None:
            cleaned  = self._ml_cleaner.batch_clean(texts)
        features = self._tfidf.transform(cleaned)
        proba    = self._ml_model.predict_proba(features)
        return self._align_proba(proba, self._ml_le.classes_)

    @timing_decorator
    def predict_dl(self, texts: List[str], cleaned: List[str] = None) -> np.ndarray:
        """
        Run BiLSTM inference. Returns (N, 3) probability array.

        Args:
            cleaned    : Pre-cleaned texts. Pass to skip redundant cleaning.
            batch_size : 256 gives ~4x fewer forward passes vs default 64,
                         reducing BiLSTM inference time by ~2x for large batches.
        """
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from src.config import DLConfig

        self.load_dl()
        if cleaned is None:
            cleaned   = self._dl_cleaner.batch_clean(texts)
        sequences = self._dl_tokenizer.texts_to_sequences(cleaned)
        padded    = pad_sequences(
            sequences,
            maxlen     = DLConfig.MAX_SEQ_LEN,
            padding    = 'post',
            truncating = 'post',
        )
        # batch_size=256: 35 forward passes for 8900 comments vs 139 at batch_size=64
        proba = self._dl_model.predict(padded, batch_size=256, verbose=0)
        return self._align_proba(proba, self._dl_le.classes_)

    def _align_proba(self, proba: np.ndarray, model_classes) -> np.ndarray:
        """Reorder probability columns to match ['negative', 'neutral', 'positive']."""
        target_order = self.LABELS
        class_list   = list(model_classes)
        idx = [class_list.index(c) for c in target_order if c in class_list]
        if len(idx) == proba.shape[1]:
            return proba[:, idx]
        return proba

    # ── Combined Prediction ────────────────────────────────────────────────
    @timing_decorator
    def predict(
        self,
        texts: List[str],
        mode : Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Predict sentiment probabilities for a list of texts.

        Args:
            texts : Raw comment texts (NOT pre-cleaned)
            mode  : 'ml', 'dl', or 'ensemble' (None = use config default)

        Returns:
            proba     : (N, 3) array — [neg, neutral, pos] probabilities
            model_used: string indicating which model(s) were used
        """
        if mode is None:
            mode = self.cfg.PRIMARY_MODEL

        if mode == 'ml':
            return self.predict_ml(texts), 'ml_ensemble'

        if mode == 'dl':
            return self.predict_dl(texts), 'dl_bilstm'

        # Ensemble: try both, blend, fall back gracefully
        ml_ok = self._ml_loaded or self._try_load_ml()
        dl_ok = self._dl_loaded or self._try_load_dl()

        if ml_ok and dl_ok:
            # ── Clean ONCE, share between ML and DL ──────────────────────
            # Without this, batch_clean() runs separately inside predict_ml
            # and predict_dl — cleaning the same N comments twice.
            shared_cleaned = self._ml_cleaner.batch_clean(texts)

            # ── Run ML and DL in PARALLEL ─────────────────────────────────
            # sklearn (TF-IDF + voting ensemble) and TensorFlow both release
            # Python's GIL during their matrix/compute operations, so threads
            # genuinely overlap here.
            # Sequential: ML(~15s) + DL(~25s) = ~40s
            # Parallel:   max(~15s, ~25s)     = ~25s  (DL is the bottleneck)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_ml = executor.submit(self.predict_ml, texts, shared_cleaned)
                future_dl = executor.submit(self.predict_dl, texts, shared_cleaned)
                ml_proba  = future_ml.result()
                dl_proba  = future_dl.result()

            combined = (
                self.cfg.ML_WEIGHT * ml_proba +
                self.cfg.DL_WEIGHT * dl_proba
            )
            return combined, 'ml+dl_ensemble'

        elif dl_ok:
            return self.predict_dl(texts), 'dl_bilstm'
        elif ml_ok:
            return self.predict_ml(texts), 'ml_ensemble'
        else:
            raise RuntimeError('No models loaded. Train models first.')

    def _try_load_ml(self) -> bool:
        try:
            self.load_ml()
            return True
        except Exception:
            return False

    def _try_load_dl(self) -> bool:
        try:
            self.load_dl()
            return True
        except Exception:
            return False

    # ── Shared aggregation helpers ────────────────────────────────────────
    def _aggregate(
        self,
        valid_comments: List[Dict],
        proba         : np.ndarray,
        model_used    : str,
        metadata      : Dict,
        elapsed       : float,
    ) -> 'VideoSentimentResult':
        """
        Build VideoSentimentResult from raw proba array + comment list.
        Used by both analyze_video() and analyze_video_pipelined().
        """
        pred_indices = proba.argmax(axis=1)
        pred_labels  = [self.LABELS[i] for i in pred_indices]
        confidences  = proba.max(axis=1)
        total        = len(pred_labels)

        # Distribution
        dist = {
            label: round(pred_labels.count(label) / total * 100, 1)
            for label in self.LABELS
        }

        # Average probability scores
        avg_scores = {
            label: float(proba[:, j].mean())
            for j, label in enumerate(self.LABELS)
        }

        # Overall sentiment = highest average probability
        overall_sentiment  = max(avg_scores, key=avg_scores.get)
        overall_confidence = avg_scores[overall_sentiment]

        # Top comments per sentiment — direct index lookup on proba array.
        # No need to build CommentPrediction objects just for this.
        def top_comments(sentiment: str, n: int = 5) -> List[str]:
            sentiment_idx = self.LABELS.index(sentiment)
            indices = [i for i, lbl in enumerate(pred_labels) if lbl == sentiment]
            indices.sort(key=lambda i: float(proba[i, sentiment_idx]), reverse=True)
            return [valid_comments[i]['text'] for i in indices[:n]]

        # Per-comment detail objects — only built for small batches.
        # For large batches (>MAX_DETAILED_COMMENTS) these objects are never
        # used by any API response field (only top_positive / top_negative are
        # returned, which we compute above without building these objects).
        # Skips ~5-8s of object creation + a third clean() pass for 19k comments.
        comment_preds = []
        if total <= self.MAX_DETAILED_COMMENTS:
            for i, comment in enumerate(valid_comments):
                scores = {
                    label: float(proba[i, j])
                    for j, label in enumerate(self.LABELS)
                }
                comment_preds.append(CommentPrediction(
                    text       = comment['text'],
                    sentiment  = pred_labels[i],
                    confidence = float(confidences[i]),
                    scores     = scores,
                ))
        else:
            logger.info(
                f'[Predictor] Skipping per-comment detail for {total} comments '
                f'(MAX_DETAILED_COMMENTS={self.MAX_DETAILED_COMMENTS})'
            )

        return VideoSentimentResult(
            video_id             = metadata.get('video_id', ''),
            video_title          = metadata.get('title', 'Unknown'),
            channel              = metadata.get('channel', 'Unknown'),
            thumbnail            = metadata.get('thumbnail', ''),
            total_comments_video = metadata.get('comment_count', 0),
            analyzed_count       = total,
            overall_sentiment    = overall_sentiment,
            overall_confidence   = overall_confidence,
            distribution         = dist,
            avg_scores           = avg_scores,
            top_positive         = top_comments('positive'),
            top_negative         = top_comments('negative'),
            processing_time_s    = elapsed,
            model_used           = model_used,
            comment_predictions  = comment_preds,
        )

    # ── Video-Level Analysis (standard) ──────────────────────────────────
    @timing_decorator
    def analyze_video(
        self,
        comments: List[Dict],
        metadata: Dict,
        mode    : Optional[str] = None,
    ) -> VideoSentimentResult:
        """
        Full video sentiment analysis — standard (non-pipelined) path.
        Used for demo/mock mode and as fallback.

        Args:
            comments : List of {'text': ..., 'likes': ..., ...} dicts
            metadata : Video metadata from youtube_fetcher
            mode     : 'ml', 'dl', or 'ensemble'
        """
        t0 = time.time()

        valid_comments = [c for c in comments if c.get('text', '').strip()]
        texts          = [c['text'] for c in valid_comments]

        if not texts:
            raise ValueError('No valid comments to analyze.')

        logger.info(f'[Predictor] Analyzing {len(texts)} comments...')

        proba, model_used = self.predict(texts, mode=mode)

        elapsed = time.time() - t0
        logger.info(
            f'[Predictor] Done in {elapsed:.2f}s | '
            f'Model: {model_used}'
        )

        return self._aggregate(valid_comments, proba, model_used, metadata, elapsed)

    # ── Video-Level Analysis (pipelined) ─────────────────────────────────
    @timing_decorator
    def analyze_video_pipelined(
        self,
        page_queue,
        metadata  : Dict,
        mode      : Optional[str] = None,
    ) -> VideoSentimentResult:
        """
        Pipelined analysis — processes each page of comments as it arrives
        from the background fetch thread, instead of waiting for all comments.

        Timeline:
          Fetch thread:  [page1][page2][page3]...[pageN][None]
          Main thread:        [predict1][predict2][predict3]...[predictN][aggregate]

          Total ≈ max(fetch_time, predict_time) instead of fetch + predict sequential.
          For 8900 comments: 62s → ~39s  (~37% faster)

        Args:
            page_queue : queue.Queue from fetch_comments_pipelined()
            metadata   : Video metadata dict
            mode       : 'ml', 'dl', or 'ensemble'
        """
        t0 = time.time()

        all_valid_comments: List[Dict]       = []
        all_proba         : List[np.ndarray] = []
        model_used        : str              = 'unknown'
        pages_processed   : int              = 0

        logger.info('[Predictor-Pipeline] Starting pipelined analysis...')

        while True:
            try:
                item = page_queue.get(timeout=120)
            except Q.Empty:
                logger.error('[Predictor-Pipeline] Timed out waiting for next page')
                break

            # None sentinel = fetch thread finished
            if item is None:
                logger.info(
                    f'[Predictor-Pipeline] Fetch complete. '
                    f'{pages_processed} pages, {len(all_valid_comments)} comments.'
                )
                break

            # Exception from fetch thread
            if isinstance(item, Exception):
                raise item

            # item is List[Dict] — one page of comments
            page_batch = [c for c in item if c.get('text', '').strip()]
            if not page_batch:
                continue

            texts = [c['text'] for c in page_batch]

            # Predict on this page immediately while next page is being fetched
            batch_proba, model_used = self.predict(texts, mode=mode)

            all_valid_comments.extend(page_batch)
            all_proba.append(batch_proba)
            pages_processed += 1

            logger.debug(
                f'[Predictor-Pipeline] Page {pages_processed}: '
                f'{len(texts)} comments | elapsed: {time.time()-t0:.1f}s'
            )

        if not all_valid_comments:
            raise ValueError('No valid comments to analyze.')

        # Stack all per-page proba arrays into one (N_total, 3) array
        proba   = np.vstack(all_proba)
        elapsed = time.time() - t0

        logger.info(
            f'[Predictor-Pipeline] Done in {elapsed:.2f}s | '
            f'Model: {model_used}'
        )

        return self._aggregate(
            all_valid_comments, proba, model_used, metadata, elapsed
        )


# ── Singleton for FastAPI ──────────────────────────────────────────────────
_predictor_instance: Optional[SentimentPredictor] = None


@timing_decorator
def get_predictor() -> SentimentPredictor:
    """FastAPI dependency injection — returns singleton predictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentPredictor()
        _predictor_instance.load_all()
    return _predictor_instance
