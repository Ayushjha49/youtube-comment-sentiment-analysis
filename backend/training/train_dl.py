"""
=============================================================================
train_dl.py — Training script for BiLSTM + Attention deep learning model
=============================================================================

USAGE:
    cd backend
    python training/train_dl.py --data ../data/processed/comments_cleaned.csv
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dl_model import DLDataPipeline, train_dl, load_dl_model
from config import DLConfig, ModelFiles, SAVE_DIR, LOG_DIR

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s [%(levelname)s] %(message)s',
    handlers= [
        logging.FileHandler(str(LOG_DIR / 'train_dl.log')),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def plot_training_history(history, save_path: str):
    """Plot accuracy + loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('BiLSTM + Attention — Training History', fontweight='bold', fontsize=14)

    epochs = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    axes[0].plot(epochs, history.history['accuracy'],     'b-o', label='Train', markersize=4)
    axes[0].plot(epochs, history.history['val_accuracy'], 'r-o', label='Val',   markersize=4)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Loss
    axes[1].plot(epochs, history.history['loss'],     'b-o', label='Train', markersize=4)
    axes[1].plot(epochs, history.history['val_loss'], 'r-o', label='Val',   markersize=4)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved training history plot → {save_path}')


def plot_training_history_inline(history):
    """For Jupyter/Colab — display inline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('BiLSTM Training History', fontweight='bold')

    axes[0].plot(history.history['accuracy'],     label='Train', color='steelblue')
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='darkorange')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train', color='steelblue')
    axes[1].plot(history.history['val_loss'], label='Val',   color='darkorange')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main(data_path: str, architecture: str = 'bilstm'):
    start = time.time()

    logger.info('=' * 70)
    logger.info(f'YouTube Sentiment — DL Training ({architecture.upper()})')
    logger.info('=' * 70)

    # ── Data pipeline ────────────────────────────────────────────────────────
    pipeline = DLDataPipeline(DLConfig)
    data     = pipeline.prepare(data_path)
    pipeline.save(str(SAVE_DIR))

    # ── Train ────────────────────────────────────────────────────────────────
    results = train_dl(data, str(SAVE_DIR), DLConfig, architecture)

    # ── Save training history plot ────────────────────────────────────────────
    plots_dir = os.path.join(str(SAVE_DIR), '..', '..', 'frontend', 'public', 'charts')
    os.makedirs(plots_dir, exist_ok=True)
    plot_training_history(
        results['history'],
        os.path.join(plots_dir, 'dl_training_history.png'),
    )

    elapsed = time.time() - start
    logger.info(f'\n[DONE] DL training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)')
    logger.info(f'[DONE] Test Accuracy: {results["test_accuracy"]:.4f}')
    logger.info(f'[DONE] Test F1:       {results["test_f1"]:.4f}')
    logger.info(f'[DONE] Model saved → {SAVE_DIR}/dl_bilstm_final.keras')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DL sentiment model')
    parser.add_argument(
        '--data',
        default='../data/processed/comments_cleaned.csv',
        help='Path to labeled dataset CSV',
    )
    parser.add_argument(
        '--arch',
        default='bilstm',
        choices=['bilstm', 'cnn_bilstm'],
        help='Architecture to train',
    )
    args = parser.parse_args()
    main(args.data, args.arch)
