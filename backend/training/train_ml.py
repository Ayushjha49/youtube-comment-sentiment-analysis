"""
=============================================================================
train_ml.py — Training script for all 5 ML models + ensemble
=============================================================================

USAGE:
    cd backend
    python training/train_ml.py --data ../data/processed/comments_cleaned.csv

EXPECTED OUTPUT (125k dataset):
    LR Ensemble Val Accuracy: ~82-86%
    Training time: ~5-15 minutes depending on hardware
"""

import os
import sys
import argparse
import logging
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score
)

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import TextCleaner, TFIDFExtractor
from ml_models import MLModelTrainer
from config import MLConfig, TFIDFConfig, ModelFiles, SAVE_DIR, LOG_DIR

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s [%(levelname)s] %(message)s',
    handlers= [
        logging.FileHandler(str(LOG_DIR / 'train_ml.log')),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_and_prepare(filepath: str):
    """Load CSV, clean text, encode labels, split data."""
    logger.info(f'Loading dataset: {filepath}')
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['comment_text', 'sentiment'])
    df = df.drop_duplicates(subset=['comment_text'])
    df['sentiment'] = df['sentiment'].str.strip().str.lower()
    df = df[df['sentiment'].isin(['positive', 'negative', 'neutral'])]

    logger.info(f'Dataset size: {len(df):,}')
    logger.info(f'Class distribution:\n{df["sentiment"].value_counts()}')

    # Text cleaning
    cleaner = TextCleaner(remove_stopwords=True)  # Enable for ML (TF-IDF)
    logger.info('Cleaning text...')
    df['cleaned_text'] = cleaner.batch_clean(df['comment_text'].tolist(), show_progress=True)
    df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)

    logger.info(f'After cleaning: {len(df):,} samples')

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df['sentiment'])
    logger.info(f'Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}')

    # Stratified splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['cleaned_text'].tolist(), y,
        test_size    = MLConfig.TEST_SIZE,
        random_state = MLConfig.RANDOM_SEED,
        stratify     = y,
    )
    val_ratio = MLConfig.VAL_SIZE / (1 - MLConfig.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size    = val_ratio,
        random_state = MLConfig.RANDOM_SEED,
        stratify     = y_temp,
    )

    # TF-IDF feature extraction
    logger.info('Fitting TF-IDF vectorizers...')
    tfidf = TFIDFExtractor(TFIDFConfig)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf   = tfidf.transform(X_val)
    X_test_tfidf  = tfidf.transform(X_test)

    logger.info(f'Feature shape: {X_train_tfidf.shape}')

    # Save vectorizers
    tfidf.save(ModelFiles.TFIDF_WORD, ModelFiles.TFIDF_CHAR)

    # Save label encoder
    with open(ModelFiles.LABEL_ENCODER, 'wb') as f:
        pickle.dump(le, f)
    logger.info(f'Saved label encoder → {ModelFiles.LABEL_ENCODER}')

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, le


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
    )
    plt.title(title, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f'Saved confusion matrix → {save_path}')


def plot_model_comparison(results: dict, save_path: str):
    """Bar chart comparing all model accuracies."""
    names    = list(results.keys())
    accs     = [results[n]['val_accuracy'] for n in names]
    f1s      = [results[n]['val_f1'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s,  width, label='F1 Score',  color='darkorange', alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_title('ML Model Comparison (Validation Set)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names])
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f'Saved model comparison → {save_path}')


def main(data_path: str, skip_knn: bool = False):
    start = time.time()

    logger.info('=' * 70)
    logger.info('YouTube Sentiment — ML Training')
    logger.info('=' * 70)

    X_train, X_val, X_test, y_train, y_val, y_test, le = load_and_prepare(data_path)

    trainer = MLModelTrainer(MLConfig)
    trainer.label_encoder = le

    if skip_knn:
        # KNN is very slow on large sparse matrices; skip if needed
        trainer.build_all()
        del trainer.models['knn']
        logger.info('[ML] Skipping KNN (use --skip_knn to always skip)')

    results = trainer.train_all(X_train, y_train, X_val, y_val)

    # ── Test set evaluation ─────────────────────────────────────────────────
    logger.info('\n' + '=' * 70)
    logger.info('FINAL TEST SET EVALUATION')
    logger.info('=' * 70)

    test_results = {}
    for name, model in trainer.models.items():
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average='weighted')
        test_results[name] = {'test_accuracy': acc, 'test_f1': f1}
        logger.info(f'{name.upper():12} Test Acc: {acc:.4f} | Test F1: {f1:.4f}')

    # ── Save all models ─────────────────────────────────────────────────────
    trainer.save_all(str(SAVE_DIR))

    # ── Plots ───────────────────────────────────────────────────────────────
    plots_dir = os.path.join(str(SAVE_DIR), '..', '..', 'frontend', 'public', 'charts')
    os.makedirs(plots_dir, exist_ok=True)

    # Model comparison chart
    plot_model_comparison(results, os.path.join(plots_dir, 'ml_model_comparison.png'))

    # Best model confusion matrix (ensemble)
    best_model = trainer.models['ensemble']
    best_preds = best_model.predict(X_test)
    plot_confusion_matrix(
        y_test, best_preds, le.classes_,
        'Ensemble Model — Confusion Matrix',
        os.path.join(plots_dir, 'ml_confusion_matrix.png'),
    )

    elapsed = time.time() - start
    logger.info(f'\n[DONE] ML training complete in {elapsed:.1f}s')
    logger.info(f'[DONE] Models saved → {SAVE_DIR}')

    return trainer, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML sentiment models')
    parser.add_argument(
        '--data',
        default='../data/processed/comments_cleaned.csv',
        help='Path to labeled dataset CSV',
    )
    parser.add_argument(
        '--skip_knn',
        action='store_true',
        help='Skip KNN training (slow on large datasets)',
    )
    args = parser.parse_args()
    main(args.data, skip_knn=args.skip_knn)
