"""
=============================================================================
config.py — Central configuration for all models, paths, and API settings
=============================================================================
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent.parent  # project root
DATA_DIR       = ROOT_DIR / 'data'
RAW_DATA       = DATA_DIR / 'raw' / 'comments_raw.csv'
PROCESSED_DATA = DATA_DIR / 'processed' / 'comments_cleaned.csv'
ML_SAVE_DIR    = ROOT_DIR / 'backend' / 'saved_ml_models'
DL_SAVE_DIR    = ROOT_DIR / 'backend' / 'saved_dl_models'
SAVE_DIR       = ML_SAVE_DIR   # backwards compat alias
LOG_DIR        = ROOT_DIR / 'logs'

ML_SAVE_DIR.mkdir(parents=True, exist_ok=True)
DL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── API ────────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY       = os.getenv('YOUTUBE_API_KEY', '')
MAX_COMMENTS          = int(os.getenv('MAX_COMMENTS', '20000'))
COMMENT_ORDER         = 'relevance'   # 'relevance' or 'time'
FETCH_TIMEOUT_SECONDS = int(os.getenv('FETCH_TIMEOUT_SECONDS', '30'))


# ── Text Preprocessing ─────────────────────────────────────────────────────
class PreprocessConfig:
    MIN_TEXT_LEN = 3
    MAX_TEXT_LEN = 500


# ── TF-IDF (for ML models) ─────────────────────────────────────────────────
class TFIDFConfig:
    # Word-level
    WORD_MAX_FEATURES = 70000       # increased: better OOV coverage for romanized words
    WORD_NGRAM_RANGE  = (1, 3)      # unigrams, bigrams, trigrams
    WORD_MIN_DF       = 2
    WORD_MAX_DF       = 0.95
    WORD_SUBLINEAR_TF = True

    # Character-level — critical for romanized/code-mixed spelling variants
    # e.g. "ramro", "raamro", "ramrooo" all meaning good/nice
    CHAR_MAX_FEATURES = 40000       # increased
    CHAR_NGRAM_RANGE  = (2, 5)
    CHAR_MIN_DF       = 3
    CHAR_MAX_DF       = 0.95
    CHAR_SUBLINEAR_TF = True

    # Combined TF-IDF weights
    WORD_WEIGHT = 0.7
    CHAR_WEIGHT = 0.3


# ── ML Model Hyperparameters ───────────────────────────────────────────────
class MLConfig:
    RANDOM_SEED = 42
    VAL_SIZE    = 0.01   # 1% val / 99% train — no test split

    # Logistic Regression
    # saga solver is faster than lbfgs for large datasets (>50k samples).
    # penalty and n_jobs removed — both deprecated in sklearn 1.8.
    LR_C        = 10.0
    LR_MAX_ITER = 2000
    LR_SOLVER   = 'saga'

    # Linear SVM
    SVM_C        = 2.0
    SVM_MAX_ITER = 3000

    # XGBoost
    # Shallower trees (depth=5) + lower LR (0.05) + more estimators (500)
    # generalizes better than fewer deep trees on sparse TF-IDF features.
    XGB_N_ESTIMATORS  = 500
    XGB_MAX_DEPTH     = 5
    XGB_LEARNING_RATE = 0.05
    XGB_SUBSAMPLE     = 0.8
    XGB_COL_SAMPLE    = 0.7


# ── Deep Learning Hyperparameters ──────────────────────────────────────────
class DLConfig:
    VOCAB_SIZE    = 60000
    MAX_SEQ_LEN   = 150
    OOV_TOKEN     = '<OOV>'
    EMBED_DIM     = 128
    LSTM_UNITS    = 128
    LSTM_DROPOUT  = 0.3
    LSTM_REC_DROP = 0.2
    DENSE_UNITS   = 64
    DROPOUT       = 0.4
    BATCH_SIZE    = 256
    EPOCHS        = 30
    LEARNING_RATE = 1e-3
    L2_LAMBDA     = 1e-4
    TEST_SIZE     = 0.15
    VAL_SIZE      = 0.15
    NUM_CLASSES   = 3
    RANDOM_SEED   = 42


# ── Inference / Prediction ─────────────────────────────────────────────────
class InferenceConfig:
    PRIMARY_MODEL        = 'ensemble'
    DL_WEIGHT            = 0.7
    ML_WEIGHT            = 0.3
    CONFIDENCE_THRESHOLD = 0.45
    LABELS               = ['negative', 'neutral', 'positive']
    LABEL_MAP            = {0: 'negative', 1: 'neutral', 2: 'positive'}


# ── Saved Model Filenames ──────────────────────────────────────────────────
class ModelFiles:
    TFIDF_WORD       = str(ML_SAVE_DIR / 'tfidf_word.pkl')
    TFIDF_CHAR       = str(ML_SAVE_DIR / 'tfidf_char.pkl')
    LABEL_ENCODER    = str(ML_SAVE_DIR / 'label_encoder.pkl')
    ML_ENSEMBLE      = str(ML_SAVE_DIR / 'ml_ensemble.pkl')
    ML_LR            = str(ML_SAVE_DIR / 'ml_lr.pkl')
    ML_SVM           = str(ML_SAVE_DIR / 'ml_svm.pkl')
    ML_XGB           = str(ML_SAVE_DIR / 'ml_xgb.pkl')
    DL_MODEL         = str(DL_SAVE_DIR / 'dl_bilstm_final.keras')
    DL_TOKENIZER     = str(DL_SAVE_DIR / 'dl_tokenizer.pkl')
    DL_LABEL_ENCODER = str(DL_SAVE_DIR / 'dl_label_encoder.pkl')
    DL_TEXT_CLEANER  = str(DL_SAVE_DIR / 'dl_text_cleaner.pkl')
