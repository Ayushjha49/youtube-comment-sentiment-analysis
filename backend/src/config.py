"""
=============================================================================
config.py — Central configuration for all models, paths, and API settings
=============================================================================
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent.parent  # Sentiment_Analysis_Final/
DATA_DIR       = ROOT_DIR / 'data'
RAW_DATA       = DATA_DIR / 'raw' / 'comments_raw.csv'
PROCESSED_DATA = DATA_DIR / 'processed' / 'comments_cleaned.csv'
ML_SAVE_DIR    = ROOT_DIR / 'backend' / 'saved_ml_models'
DL_SAVE_DIR    = ROOT_DIR / 'backend' / 'saved_dl_models'
SAVE_DIR       = ML_SAVE_DIR   # keep this for backwards compat with training scripts
LOG_DIR        = ROOT_DIR / 'logs'

ML_SAVE_DIR.mkdir(parents=True, exist_ok=True)
DL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── API ────────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY   = os.getenv('YOUTUBE_API_KEY', '')
MAX_COMMENTS      = int(os.getenv('MAX_COMMENTS', '10000'))  # reads from .env
COMMENT_ORDER     = 'relevance'  # 'relevance' or 'time'

# ── Fetch time limit ────────────────────────────────────────────────────────
# Fetch as many comments as possible within this many seconds, then stop
# and analyze whatever was collected. Set to 0 to disable (fetch until
# MAX_COMMENTS is reached). Change via .env: FETCH_TIMEOUT_SECONDS=120
FETCH_TIMEOUT_SECONDS = int(os.getenv('FETCH_TIMEOUT_SECONDS', '30'))


# ── Text Preprocessing ─────────────────────────────────────────────────────
class PreprocessConfig:
    MIN_TEXT_LEN  = 3          # Ignore comments shorter than this
    MAX_TEXT_LEN  = 500        # Truncate comments longer than this


# ── TF-IDF (for ML models) ─────────────────────────────────────────────────
class TFIDFConfig:
    # Word-level
    WORD_MAX_FEATURES  = 50000
    WORD_NGRAM_RANGE   = (1, 3)   # Unigrams, bigrams, trigrams
    WORD_MIN_DF        = 2
    WORD_MAX_DF        = 0.95
    WORD_SUBLINEAR_TF  = True

    # Character-level (great for romanized/code-mixed text)
    CHAR_MAX_FEATURES  = 30000
    CHAR_NGRAM_RANGE   = (2, 5)   # char n-grams
    CHAR_MIN_DF        = 3
    CHAR_MAX_DF        = 0.95
    CHAR_SUBLINEAR_TF  = True

    # Combined TF-IDF weight
    WORD_WEIGHT        = 0.7
    CHAR_WEIGHT        = 0.3


# ── ML Model Hyperparameters ───────────────────────────────────────────────
class MLConfig:
    RANDOM_SEED   = 42
    TEST_SIZE     = 0.15
    VAL_SIZE      = 0.15
    N_JOBS        = -1         # Use all CPU cores

    # Logistic Regression
    LR_C          = 5.0
    LR_MAX_ITER   = 1000
    LR_SOLVER     = 'lbfgs'
    LR_PENALTY    = 'l2'

    # Random Forest
    RF_N_ESTIMATORS    = 300
    RF_MAX_DEPTH       = None
    RF_MIN_SAMPLES     = 2
    RF_MAX_FEATURES    = 'sqrt'

    # Linear SVM
    SVM_C         = 1.0
    SVM_MAX_ITER  = 2000

    # KNN
    KNN_K         = 11
    KNN_METRIC    = 'cosine'
    KNN_WEIGHTS   = 'distance'

    # Gradient Boosting (XGBoost)
    XGB_N_ESTIMATORS   = 300
    XGB_MAX_DEPTH      = 6
    XGB_LEARNING_RATE  = 0.1
    XGB_SUBSAMPLE      = 0.8
    XGB_COL_SAMPLE     = 0.8

    # Ensemble voting weights: LR, SVM, XGB (best 3 for text)
    ENSEMBLE_WEIGHTS   = [2, 2, 1]   # LR and SVM get higher weight


# ── Deep Learning Hyperparameters ──────────────────────────────────────────
class DLConfig:
    # Tokenizer
    VOCAB_SIZE      = 60000
    MAX_SEQ_LEN     = 150
    OOV_TOKEN       = '<OOV>'

    # Embedding
    EMBED_DIM       = 128

    # BiLSTM
    LSTM_UNITS      = 128
    LSTM_DROPOUT    = 0.3
    LSTM_REC_DROP   = 0.2

    # Dense
    DENSE_UNITS     = 64
    DROPOUT         = 0.4

    # Training
    BATCH_SIZE      = 256
    EPOCHS          = 30
    LEARNING_RATE   = 1e-3
    L2_LAMBDA       = 1e-4

    # Data splits
    TEST_SIZE       = 0.15
    VAL_SIZE        = 0.15
    NUM_CLASSES     = 3
    RANDOM_SEED     = 42


# ── Inference / Prediction ─────────────────────────────────────────────────
class InferenceConfig:
    # Which model to use for production
    PRIMARY_MODEL   = 'ensemble'   # 'ml', 'dl', or 'ensemble'

    # DL + ML ensemble blending (only used when PRIMARY_MODEL = 'ensemble')
    DL_WEIGHT       = 0.6
    ML_WEIGHT       = 0.4

    # Confidence threshold — below this, report as 'uncertain'
    CONFIDENCE_THRESHOLD = 0.45

    # Sentiment labels
    LABELS          = ['negative', 'neutral', 'positive']
    LABEL_MAP       = {0: 'negative', 1: 'neutral', 2: 'positive'}


# ── Saved Model Filenames ──────────────────────────────────────────────────
class ModelFiles:
    TFIDF_WORD        = str(ML_SAVE_DIR / 'tfidf_word.pkl')
    TFIDF_CHAR        = str(ML_SAVE_DIR / 'tfidf_char.pkl')
    LABEL_ENCODER     = str(ML_SAVE_DIR / 'label_encoder.pkl')
    ML_ENSEMBLE       = str(ML_SAVE_DIR / 'ml_ensemble.pkl')
    ML_LR             = str(ML_SAVE_DIR / 'ml_lr.pkl')
    ML_RF             = str(ML_SAVE_DIR / 'ml_rf.pkl')
    ML_SVM            = str(ML_SAVE_DIR / 'ml_svm.pkl')
    ML_KNN            = str(ML_SAVE_DIR / 'ml_knn.pkl')
    ML_XGB            = str(ML_SAVE_DIR / 'ml_xgb.pkl')
    DL_MODEL          = str(DL_SAVE_DIR / 'dl_bilstm_final.keras')
    DL_TOKENIZER      = str(DL_SAVE_DIR / 'dl_tokenizer.pkl')
    DL_LABEL_ENCODER  = str(DL_SAVE_DIR / 'dl_label_encoder.pkl')
    DL_TEXT_CLEANER   = str(DL_SAVE_DIR / 'dl_text_cleaner.pkl')
