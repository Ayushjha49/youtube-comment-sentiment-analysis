"""
=============================================================================
predictor.py — Unified inference engine for ML + DL sentiment prediction
=============================================================================

PREDICTION FLOW:
  1. Raw comment texts  →  TextCleaner  →  Cleaned texts
  2a. Cleaned texts  →  TF-IDF  →  ML Ensemble  →  probabilities
  2b. Cleaned texts  →  Tokenize/Pad  →  BiLSTM  →  probabilities
  3. Weighted blend of ML + DL probabilities  →  Final predictions
  4. Aggregate per-comment predictions  →  Video-level summary

CACHING:
  Models are loaded once and kept in memory for fast repeated inference.
"""

import os
import sys
import pickle
import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger(__name__)


@dataclass
class CommentPrediction:
    text      : str
    cleaned   : str
    sentiment : str   # 'positive', 'negative', 'neutral'
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
    distribution        : Dict[str, float]  # percentages
    avg_scores          : Dict[str, float]  # average probabilities
    top_positive        : List[str]
    top_negative        : List[str]
    processing_time_s   : float
    model_used          : str
    comment_predictions : List[CommentPrediction] = field(default_factory=list)


class SentimentPredictor:
    """
    Unified predictor that loads and runs ML and/or DL models.
    
    Lazy-loads models on first use and caches them.
    Thread-safe for FastAPI concurrent requests (models are read-only after load).
    """

    LABELS = ['negative', 'neutral', 'positive']

    def __init__(self, config=None):
        if config is None:
            from src.config import InferenceConfig, ModelFiles
            self.cfg   = InferenceConfig
            self.files = ModelFiles
        else:
            self.cfg   = config['inference']
            self.files = config['files']

        self._ml_model       = None
        self._tfidf          = None
        self._ml_le          = None
        self._dl_model       = None
        self._dl_tokenizer   = None
        self._dl_cleaner     = None
        self._dl_le          = None
        self._ml_cleaner     = None

        self._ml_loaded = False
        self._dl_loaded = False

    # ── Model Loading ─────────────────────────────────────────────────────
    def load_ml(self):
        """Load ML ensemble + TF-IDF vectorizers."""
        if self._ml_loaded:
            return
        logger.info('[Predictor] Loading ML ensemble...')
        try:
            from src.preprocess import TFIDFExtractor, TextCleaner
            self._tfidf    = TFIDFExtractor.load(self.files.TFIDF_WORD, self.files.TFIDF_CHAR)
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
    def predict_ml(self, texts: List[str]) -> np.ndarray:
        """Run ML ensemble inference. Returns (N, 3) probability array."""
        self.load_ml()
        cleaned  = self._ml_cleaner.batch_clean(texts)
        features = self._tfidf.transform(cleaned)
        proba    = self._ml_model.predict_proba(features)

        # Align label order to ['negative', 'neutral', 'positive']
        return self._align_proba(proba, self._ml_le.classes_)

    def predict_dl(self, texts: List[str]) -> np.ndarray:
        """Run BiLSTM inference. Returns (N, 3) probability array."""
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from src.config import DLConfig

        self.load_dl()
        cleaned   = self._dl_cleaner.batch_clean(texts)
        sequences = self._dl_tokenizer.texts_to_sequences(cleaned)
        padded    = pad_sequences(
            sequences,
            maxlen     = DLConfig.MAX_SEQ_LEN,
            padding    = 'post',
            truncating = 'post',
        )
        proba = self._dl_model.predict(padded, batch_size=64, verbose=0)

        return self._align_proba(proba, self._dl_le.classes_)

    def _align_proba(self, proba: np.ndarray, model_classes) -> np.ndarray:
        """Reorder probability columns to match ['negative', 'neutral', 'positive']."""
        target_order = self.LABELS  # ['negative', 'neutral', 'positive']
        class_list   = list(model_classes)
        idx = [class_list.index(c) for c in target_order if c in class_list]
        if len(idx) == proba.shape[1]:
            return proba[:, idx]
        return proba

    # ── Combined Prediction ────────────────────────────────────────────────
    def predict(
        self,
        texts: List[str],
        mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Predict sentiment probabilities for a list of texts.

        Args:
            texts : Raw comment texts (NOT pre-cleaned)
            mode  : 'ml', 'dl', or 'ensemble' (None = use config default)

        Returns:
            proba     : (N, 3) array of [neg, neutral, pos] probabilities
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
            ml_proba = self.predict_ml(texts)
            dl_proba = self.predict_dl(texts)
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
        except:
            return False

    def _try_load_dl(self) -> bool:
        try:
            self.load_dl()
            return True
        except:
            return False

    # ── Video-Level Analysis ──────────────────────────────────────────────
    def analyze_video(
        self,
        comments: List[Dict],
        metadata: Dict,
        mode: Optional[str] = None,
    ) -> VideoSentimentResult:
        """
        Full video sentiment analysis.

        Args:
            comments: List of {'text': ..., 'likes': ..., ...} dicts
            metadata: Video metadata from youtube_fetcher
            mode    : 'ml', 'dl', or 'ensemble'

        Returns:
            VideoSentimentResult with all statistics
        """
        t0 = time.time()

        # Filter out empty comments
        valid_comments = [c for c in comments if c.get('text', '').strip()]
        texts          = [c['text'] for c in valid_comments]

        if not texts:
            raise ValueError('No valid comments to analyze.')

        logger.info(f'[Predictor] Analyzing {len(texts)} comments...')

        # Run inference
        proba, model_used = self.predict(texts, mode=mode)
        pred_indices      = proba.argmax(axis=1)
        pred_labels       = [self.LABELS[i] for i in pred_indices]
        confidences       = proba.max(axis=1)

        # Per-comment predictions
        comment_preds = []
        from src.preprocess import TextCleaner
        cleaner = TextCleaner()
        for i, comment in enumerate(valid_comments):
            raw    = comment['text']
            scores = {label: float(proba[i, j]) for j, label in enumerate(self.LABELS)}
            comment_preds.append(CommentPrediction(
                text       = raw,
                cleaned    = cleaner.clean(raw),
                sentiment  = pred_labels[i],
                confidence = float(confidences[i]),
                scores     = scores,
            ))

        # Distribution
        total = len(pred_labels)
        dist  = {
            label: round(pred_labels.count(label) / total * 100, 1)
            for label in self.LABELS
        }

        # Average probability scores
        avg_scores = {label: float(proba[:, j].mean()) for j, label in enumerate(self.LABELS)}

        # Overall sentiment = highest average probability
        overall_sentiment  = max(avg_scores, key=avg_scores.get)
        overall_confidence = avg_scores[overall_sentiment]

        # Top comments for each sentiment
        def top_comments(sentiment: str, n: int = 5) -> List[str]:
            candidates = [
                (cp.text, cp.scores[sentiment])
                for cp in comment_preds
                if cp.sentiment == sentiment
            ]
            candidates.sort(key=lambda x: -x[1])
            return [text for text, _ in candidates[:n]]

        elapsed = time.time() - t0
        logger.info(f'[Predictor] Done in {elapsed:.2f}s | Overall: {overall_sentiment} ({overall_confidence:.1%})')

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



    # ── Pipelined Video Analysis ──────────────────────────────────────────
    def analyze_video_pipelined(
        self,
        page_queue,           # queue.Queue from fetch_comments_pipelined()
        metadata : Dict,
        mode     : Optional[str] = None,
    ) -> 'VideoSentimentResult':
        """
        Pipelined analysis: process each page of comments as it arrives
        from the background fetch thread, instead of waiting for ALL
        comments to be fetched first.

        Timeline:
          Fetch thread:  [page1][page2][page3]...[pageN][None]
          Main thread:        [predict1][predict2][predict3]...[predictN][aggregate]

          Total ≈ max(fetch_time, predict_time) instead of fetch + predict

        For 8900 comments: 62s → ~39s  (~37% faster)
        """
        import queue as Q

        t0 = time.time()

        all_valid_comments : List[Dict]       = []
        all_proba          : List[np.ndarray] = []
        model_used         : str              = 'unknown'
        pages_processed    : int              = 0

        logger.info('[Predictor-Pipeline] Starting pipelined analysis...')

        while True:
            try:
                # Block until next page arrives (timeout prevents hanging forever)
                item = page_queue.get(timeout=120)
            except Q.Empty:
                logger.error('[Predictor-Pipeline] Timed out waiting for next page')
                break

            # None sentinel = fetch thread is done
            if item is None:
                logger.info(f'[Predictor-Pipeline] Fetch complete. '
                            f'Processed {pages_processed} pages, '
                            f'{len(all_valid_comments)} comments total.')
                break

            # Error from fetch thread
            if isinstance(item, Exception):
                raise item

            # item is a List[Dict] — one page of comments
            page_batch = [c for c in item if c.get('text', '').strip()]
            if not page_batch:
                continue

            texts = [c['text'] for c in page_batch]

            # Predict on this page batch immediately
            batch_proba, model_used = self.predict(texts, mode=mode)

            all_valid_comments.extend(page_batch)
            all_proba.append(batch_proba)
            pages_processed += 1

            logger.debug(f'[Predictor-Pipeline] Page {pages_processed} done: '
                         f'{len(texts)} comments | '
                         f'elapsed: {time.time()-t0:.1f}s')

        if not all_valid_comments:
            raise ValueError('No valid comments to analyze.')

        # ── Aggregate all batch results ───────────────────────────────────
        proba        = np.vstack(all_proba)   # stack all (N_i, 3) arrays → (N_total, 3)
        pred_indices = proba.argmax(axis=1)
        pred_labels  = [self.LABELS[i] for i in pred_indices]
        confidences  = proba.max(axis=1)

        # Per-comment predictions
        from src.preprocess import TextCleaner
        cleaner       = TextCleaner()
        comment_preds = []
        for i, comment in enumerate(all_valid_comments):
            raw    = comment['text']
            scores = {label: float(proba[i, j]) for j, label in enumerate(self.LABELS)}
            comment_preds.append(CommentPrediction(
                text       = raw,
                cleaned    = cleaner.clean(raw),
                sentiment  = pred_labels[i],
                confidence = float(confidences[i]),
                scores     = scores,
            ))

        total      = len(pred_labels)
        dist       = {
            label: round(pred_labels.count(label) / total * 100, 1)
            for label in self.LABELS
        }
        avg_scores = {label: float(proba[:, j].mean()) for j, label in enumerate(self.LABELS)}

        overall_sentiment  = max(avg_scores, key=avg_scores.get)
        overall_confidence = avg_scores[overall_sentiment]

        def top_comments(sentiment: str, n: int = 5) -> List[str]:
            candidates = [
                (cp.text, cp.scores[sentiment])
                for cp in comment_preds
                if cp.sentiment == sentiment
            ]
            candidates.sort(key=lambda x: -x[1])
            return [text for text, _ in candidates[:n]]

        elapsed = time.time() - t0
        logger.info(
            f'[Predictor-Pipeline] Done in {elapsed:.2f}s | '
            f'Overall: {overall_sentiment} ({overall_confidence:.1%}) | '
            f'Model: {model_used}'
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

# ── Singleton for FastAPI ──────────────────────────────────────────────────
_predictor_instance: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    """FastAPI dependency injection — returns singleton predictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentPredictor()
        _predictor_instance.load_all()
    return _predictor_instance
