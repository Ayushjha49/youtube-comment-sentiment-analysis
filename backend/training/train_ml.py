"""
=============================================================================
train_ml.py — ML Ensemble Training: LR + SVM + XGBoost
=============================================================================

USAGE:
    cd backend
    python training/train_ml.py

    # Custom data path:
    python training/train_ml.py --data ../data/processed/comments_cleaned.csv

WHAT THIS SCRIPT DOES:
    1. Loads and cleans the dataset using the same TextCleaner used in production
    2. Fits TF-IDF vectorizers (word + char) on 99% of the data
    3. Trains LR, SVM, and XGBoost individually
    4. Auto-tunes ensemble weights by searching 27 combinations on the val set
    5. Builds and saves the final VotingEnsemble
    6. Saves all models + vectorizers to saved_ml_models/
    7. Saves charts to frontend/public/charts/

SPLIT STRATEGY:
    99% train / 1% val / 0% test
    Real-world testing happens on actual YouTube comments in production.
    The 1% val set is used only for ensemble weight tuning — the models
    never fit to it, so there is no data leakage.

OUTPUT FILES (saved_ml_models/):
    tfidf_word.pkl      — word-level TF-IDF vectorizer
    tfidf_char.pkl      — character-level TF-IDF vectorizer
    label_encoder.pkl   — label encoder (negative/neutral/positive → 0/1/2)
    ml_lr.pkl           — Logistic Regression
    ml_svm.pkl          — Linear SVM (calibrated)
    ml_xgb.pkl          — XGBoost
    ml_ensemble.pkl     — VotingEnsemble (LR + SVM + XGB, auto-tuned weights)
=============================================================================
"""

import os
import sys
import argparse
import logging
import pickle
import time
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import TextCleaner, TFIDFExtractor
from ml_models import MLModelTrainer
from config import MLConfig, TFIDFConfig, ModelFiles, SAVE_DIR, LOG_DIR, ROOT_DIR


# ── Logging ────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    log_file = str(LOG_DIR / 'train_ml.log')
    handlers = [
        RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level   = logging.INFO,
        format  = '%(asctime)s [%(levelname)s] %(message)s',
        handlers= handlers,
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# =============================================================================
# DATA
# =============================================================================

def load_and_prepare(data_path: str):
    """
    Load dataset, clean with the production TextCleaner, encode labels,
    split 99/1, fit TF-IDF, and save vectorizers + label encoder.

    Using the same TextCleaner here as in production ensures that the
    TF-IDF vocabulary built during training exactly matches the features
    the model receives at inference time.
    """
    logger.info(f'Loading dataset: {data_path}')
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['comment_text', 'sentiment'])
    df = df.drop_duplicates(subset=['comment_text'])
    df['sentiment'] = df['sentiment'].str.strip().str.lower()
    df = df[df['sentiment'].isin(['positive', 'negative', 'neutral'])].reset_index(drop=True)

    logger.info(f'Samples: {len(df):,}')
    logger.info(f'Distribution:\n{df["sentiment"].value_counts()}')

    # ── Clean text ─────────────────────────────────────────────────────────
    # remove_stopwords=True: TF-IDF benefits from a cleaner, denser vocabulary
    cleaner = TextCleaner(remove_stopwords=True)
    logger.info('Cleaning text...')
    t0 = time.time()
    df['cleaned_text'] = cleaner.batch_clean(df['comment_text'].tolist(), show_progress=True)
    df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
    logger.info(f'Cleaned in {time.time()-t0:.1f}s — {len(df):,} samples remaining')

    # ── Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(df['sentiment'])
    logger.info(f'Label map: {dict(zip(le.classes_, le.transform(le.classes_)))}')

    # ── 99% train / 1% val ────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        df['cleaned_text'].tolist(), y,
        test_size    = MLConfig.VAL_SIZE,
        random_state = MLConfig.RANDOM_SEED,
        stratify     = y,
    )
    logger.info(f'Train: {len(X_train):,} | Val: {len(X_val):,}')

    # ── TF-IDF ────────────────────────────────────────────────────────────
    logger.info('Fitting TF-IDF vectorizers...')
    t0    = time.time()
    tfidf = TFIDFExtractor(TFIDFConfig)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf   = tfidf.transform(X_val)
    logger.info(f'TF-IDF done in {time.time()-t0:.1f}s — shape: {X_train_tfidf.shape}')

    # Save vectorizers and label encoder to production model directory
    tfidf.save(ModelFiles.TFIDF_WORD, ModelFiles.TFIDF_CHAR)
    with open(ModelFiles.LABEL_ENCODER, 'wb') as f:
        pickle.dump(le, f)
    logger.info(f'Saved tfidf_word.pkl, tfidf_char.pkl, label_encoder.pkl → {SAVE_DIR}')

    return X_train_tfidf, X_val_tfidf, y_train, y_val, le


# =============================================================================
# PLOTS
# =============================================================================

def _charts_dir() -> str:
    """
    Return the charts output directory.
    Uses ROOT_DIR from config — no fragile relative path traversal.
    """
    charts = str(ROOT_DIR / 'frontend' / 'public' / 'charts')
    os.makedirs(charts, exist_ok=True)
    return charts


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    cm = __import__('sklearn.metrics', fromlist=['confusion_matrix']).confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title('Ensemble — Confusion Matrix (Val Set)', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved → {save_path}')


def plot_model_comparison(results: dict, save_path: str):
    names = list(results.keys())
    accs  = [results[n]['val_accuracy'] for n in names]
    f1s   = [results[n]['val_f1']       for n in names]
    x     = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - 0.175, accs, 0.35, label='Accuracy', color='steelblue',  alpha=0.85)
    b2 = ax.bar(x + 0.175, f1s,  0.35, label='F1 Score',  color='darkorange', alpha=0.85)
    ax.set_title('ML Model Comparison (Val Set)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved → {save_path}')


# =============================================================================
# MAIN
# =============================================================================

def main(data_path: str):
    start = time.time()

    logger.info('=' * 70)
    logger.info('YouTube Sentiment — ML Training (LR + SVM + XGBoost)')
    logger.info('=' * 70)
    logger.info(f'Data path : {data_path}')
    logger.info(f'Save dir  : {SAVE_DIR}')
    logger.info(f'Split     : 99% train / 1% val / 0% test')

    # ── Data + TF-IDF ─────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val, le = load_and_prepare(data_path)

    # ── Train models + auto-tune ensemble ─────────────────────────────────
    trainer = MLModelTrainer(MLConfig)
    trainer.label_encoder = le
    results = trainer.train_all(X_train, y_train, X_val, y_val)

    # ── Save models ────────────────────────────────────────────────────────
    trainer.save_all(str(SAVE_DIR))

    # ── Classification report on val set ──────────────────────────────────
    ens_preds = trainer.models['ensemble'].predict(X_val)
    logger.info('\nClassification Report (Ensemble, Val Set):')
    logger.info('\n' + classification_report(y_val, ens_preds, target_names=le.classes_, digits=4))

    # ── Plots ──────────────────────────────────────────────────────────────
    charts = _charts_dir()
    plot_model_comparison(results, os.path.join(charts, 'ml_model_comparison.png'))
    plot_confusion_matrix(
        y_val, ens_preds, list(le.classes_),
        os.path.join(charts, 'ml_confusion_matrix.png'),
    )

    elapsed = time.time() - start
    logger.info(f'\n[DONE] Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)')
    logger.info(f'[DONE] Models saved → {SAVE_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML sentiment models')
    parser.add_argument(
        '--data',
        default='../data/processed/comments_cleaned.csv',
        help='Path to labeled dataset CSV',
    )
    args = parser.parse_args()
    main(args.data)
