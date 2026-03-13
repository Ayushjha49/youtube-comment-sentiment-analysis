"""
=============================================================================
ml_models.py — ML Model Builders + Trainer
=============================================================================

MODELS:
  1. Logistic Regression  — Fast, strong baseline; great regularization on sparse TF-IDF
  2. Linear SVM           — State-of-art for high-dimensional text; margin maximization
  3. XGBoost              — Gradient boosting; captures non-linear feature interactions
  4. VotingEnsemble       — Soft voting over LR + SVM + XGB with auto-tuned weights

WHY NOT KNN OR RANDOM FOREST?
  KNN  — cosine similarity on TF-IDF is noisy; slow at inference (scans all training points)
  RF   — overconfident on sparse features; adds noise to soft voting probability averaging

EXPECTED ACCURACY (125k dataset, 3 classes):
  LR       : ~78-82%
  SVM      : ~80-84%
  XGB      : ~78-83%
  ENSEMBLE : ~82-86%  ← best ML performance
  BiLSTM   : ~85-90%  ← DL wins on romanized/code-mixed text
"""

import os
import pickle
import logging
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False
    logger.warning('[ML] XGBoost not found — falling back to sklearn GradientBoosting')


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_logistic_regression(config) -> LogisticRegression:
    """
    Best overall text classifier for high-dimensional TF-IDF features.
    saga solver is significantly faster than lbfgs for large datasets (>50k samples).
    Note: penalty and n_jobs intentionally omitted — both deprecated in sklearn 1.8.
    """
    return LogisticRegression(
        C            = config.LR_C,
        max_iter     = config.LR_MAX_ITER,
        solver       = config.LR_SOLVER,
        class_weight = 'balanced',
        random_state = config.RANDOM_SEED,
    )


def build_svm(config) -> CalibratedClassifierCV:
    """
    Linear SVM — consistently strong for text classification.
    Wrapped in CalibratedClassifierCV for probability output via Platt scaling.
    LinearSVC does not have predict_proba natively.
    """
    base = LinearSVC(
        C            = config.SVM_C,
        max_iter     = config.SVM_MAX_ITER,
        class_weight = 'balanced',
        random_state = config.RANDOM_SEED,
    )
    return CalibratedClassifierCV(base, cv=3, method='sigmoid')


def build_xgboost(config):
    """
    XGBoost gradient boosting — captures non-linear feature interactions.
    Uses histogram-based tree method for speed on high-dimensional sparse data.
    Shallow trees (depth=5) + low LR (0.05) generalizes better on sparse TF-IDF.
    """
    if XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators     = config.XGB_N_ESTIMATORS,
            max_depth        = config.XGB_MAX_DEPTH,
            learning_rate    = config.XGB_LEARNING_RATE,
            subsample        = config.XGB_SUBSAMPLE,
            colsample_bytree = config.XGB_COL_SAMPLE,
            eval_metric      = 'mlogloss',
            random_state     = config.RANDOM_SEED,
            tree_method      = 'hist',
            n_jobs           = -1,
            verbosity        = 0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators  = config.XGB_N_ESTIMATORS,
            max_depth     = config.XGB_MAX_DEPTH,
            learning_rate = config.XGB_LEARNING_RATE,
            subsample     = config.XGB_SUBSAMPLE,
            random_state  = config.RANDOM_SEED,
        )


def build_voting_ensemble(lr, svm, xgb, weights: List[int]) -> VotingClassifier:
    """
    Soft voting ensemble over LR + SVM + XGB.
    Soft voting averages class probabilities — more stable than hard voting.
    Weights are auto-tuned on the validation set by tune_ensemble_weights().
    """
    return VotingClassifier(
        estimators = [('lr', lr), ('svm', svm), ('xgb', xgb)],
        voting     = 'soft',
        weights    = weights,
        n_jobs     = 1,   # sub-models already parallelize internally
    )


# =============================================================================
# WEIGHT TUNING
# =============================================================================

def tune_ensemble_weights(
    lr, svm, xgb,
    X_val, y_val,
) -> Tuple[List[int], float]:
    """
    Search 27 integer weight combinations on the validation set to find
    the optimal LR / SVM / XGB blend. Returns (best_weights, best_accuracy).

    Why not cross-validation?
    Weight tuning is a fast grid search over 3 scalars — CV would be overkill
    and would require re-fitting all sub-models multiple times.
    """
    lr_p  = lr.predict_proba(X_val)
    svm_p = svm.predict_proba(X_val)
    xgb_p = xgb.predict_proba(X_val)

    best_acc, best_weights = 0.0, [2, 2, 1]

    for w_lr, w_svm, w_xgb in product([1, 2, 3], repeat=3):
        w_total = w_lr + w_svm + w_xgb
        blended = (w_lr * lr_p + w_svm * svm_p + w_xgb * xgb_p) / w_total
        acc     = accuracy_score(y_val, blended.argmax(axis=1))
        if acc > best_acc:
            best_acc, best_weights = acc, [w_lr, w_svm, w_xgb]

    logger.info(
        f'[ML] Best ensemble weights — '
        f'LR={best_weights[0]}, SVM={best_weights[1]}, XGB={best_weights[2]} '
        f'| Val Acc: {best_acc:.4f}'
    )
    return best_weights, best_acc


# =============================================================================
# TRAINER
# =============================================================================

class MLModelTrainer:
    """
    Builds, trains, evaluates, and saves LR + SVM + XGBoost + VotingEnsemble.
    """

    def __init__(self, config=None):
        if config is None:
            from config import MLConfig
            config = MLConfig
        self.cfg           = config
        self.models: Dict  = {}
        self.label_encoder = LabelEncoder()

    def build_all(self) -> Dict:
        """Instantiate all three model objects."""
        self.models = {
            'lr'  : build_logistic_regression(self.cfg),
            'svm' : build_svm(self.cfg),
            'xgb' : build_xgboost(self.cfg),
        }
        logger.info(f'[ML] Models built: {list(self.models.keys())}')
        return self.models

    def _train_one(self, name: str, model, X_train, y_train, X_val, y_val) -> Dict:
        """Train a single model and return its val metrics."""
        import time
        logger.info(f'[ML] Training {name.upper()}...')
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        preds = model.predict(X_val)
        acc   = accuracy_score(y_val, preds)
        f1    = f1_score(y_val, preds, average='weighted')
        logger.info(f'[ML] {name.upper()} — {elapsed:.1f}s | Val Acc: {acc:.4f} | Val F1: {f1:.4f}')
        return {'val_accuracy': acc, 'val_f1': f1}

    def train_all(self, X_train, y_train, X_val, y_val) -> Dict:
        """
        Train LR, SVM, XGB individually, auto-tune ensemble weights,
        then build and train the VotingEnsemble.
        Returns metrics dict for all models including ensemble.
        """
        if not self.models:
            self.build_all()

        results = {}
        for name, model in self.models.items():
            results[name] = self._train_one(name, model, X_train, y_train, X_val, y_val)

        # Auto-tune ensemble weights on val set
        best_weights, _ = tune_ensemble_weights(
            self.models['lr'], self.models['svm'], self.models['xgb'],
            X_val, y_val,
        )

        # Build and train VotingEnsemble
        # sklearn VotingClassifier requires fit() even with pre-trained estimators
        logger.info(f'[ML] Building VotingEnsemble (weights={best_weights})...')
        ensemble = build_voting_ensemble(
            self.models['lr'], self.models['svm'], self.models['xgb'],
            weights=best_weights,
        )
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble

        ens_preds = ensemble.predict(X_val)
        results['ensemble'] = {
            'val_accuracy': accuracy_score(y_val, ens_preds),
            'val_f1'      : f1_score(y_val, ens_preds, average='weighted'),
        }

        # Summary table
        logger.info('\n' + '=' * 55)
        logger.info('MODEL COMPARISON (Val Set):')
        logger.info('=' * 55)
        logger.info(f'{"Model":<12} {"Accuracy":>10} {"F1":>10}')
        logger.info('-' * 35)
        best_val = max(v['val_accuracy'] for v in results.values())
        for name, m in sorted(results.items(), key=lambda x: -x[1]['val_accuracy']):
            marker = ' ← best' if m['val_accuracy'] == best_val else ''
            logger.info(f'{name.upper():<12} {m["val_accuracy"]:>10.4f} {m["val_f1"]:>10.4f}{marker}')
        logger.info('=' * 55)

        return results

    def save_all(self, save_dir: str):
        """Save all trained models to disk."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(save_dir, f'ml_{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f'[ML] Saved {name} → {path}')

    @staticmethod
    def load_model(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
