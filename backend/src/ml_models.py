"""
=============================================================================
ml_models.py — 5 Supervised ML Models + Voting Ensemble
=============================================================================

MODELS:
  1. Logistic Regression  — Fast, strong baseline for text; interpretable
  2. Random Forest        — Handles non-linear patterns, robust to noise
  3. Linear SVM           — Best for high-dimensional text (TF-IDF) spaces
  4. KNN                  — Simple, no training; slow at inference on large data
  5. XGBoost              — Gradient boosting; best accuracy on tabular/sparse
  6. VotingEnsemble       — Soft voting over LR + SVM + XGB (best 3)

WHY THESE FIVE?
  • LR   — Extremely fast, great L2 regularization for sparse features
  • RF   — Decorrelated trees handle romanized text variance well
  • SVM  — State-of-art for high-dim text; margin maximization
  • KNN  — Non-parametric; good reference point for comparison
  • XGB  — Boosting captures complex feature interactions

ENSEMBLE STRATEGY:
  Soft voting (average probabilities) over LR + SVM + XGB.
  KNN and RF are excluded from ensemble — KNN is too slow,
  RF tends to be overconfident on code-mixed data.

EXPECTED ACCURACY (125k dataset, 3 classes):
  LR  : ~78-82%
  SVM : ~80-84%
  RF  : ~74-79%
  KNN : ~68-73%
  XGB : ~78-83%
  ENS : ~82-86%  ← best ML performance
  BiLSTM: ~85-90%  ← DL wins on romanized/code-mixed text
"""

import os
import sys
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ── Try importing XGBoost, fallback to GradientBoosting ───────────────────
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    logger.info('[ML] XGBoost available')
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False
    logger.warning('[ML] XGBoost not found, using sklearn GradientBoosting')


# =============================================================================
# INDIVIDUAL MODEL BUILDERS
# =============================================================================
def build_logistic_regression(config) -> LogisticRegression:
    """
    Best overall text classifier for high-dimensional TF-IDF features.
    L2 regularization prevents overfitting on rare romanized words.
    """
    return LogisticRegression(
        C           = config.LR_C,
        max_iter    = config.LR_MAX_ITER,
        solver      = config.LR_SOLVER,
        penalty     = config.LR_PENALTY,
        class_weight= 'balanced',
        random_state= config.RANDOM_SEED,
        n_jobs      = config.N_JOBS,
    )


def build_random_forest(config) -> RandomForestClassifier:
    """
    Ensemble of decorrelated decision trees.
    Good at capturing non-linear word co-occurrence patterns.
    """
    return RandomForestClassifier(
        n_estimators    = config.RF_N_ESTIMATORS,
        max_depth       = config.RF_MAX_DEPTH,
        min_samples_split= config.RF_MIN_SAMPLES,
        max_features    = config.RF_MAX_FEATURES,
        class_weight    = 'balanced',
        random_state    = config.RANDOM_SEED,
        n_jobs          = config.N_JOBS,
        verbose         = 1,
    )


def build_svm(config):
    """
    Linear SVM — consistently best for text classification.
    CalibratedClassifierCV wraps it to produce probability estimates
    (LinearSVC doesn't have predict_proba natively).
    """
    base_svm = LinearSVC(
        C           = config.SVM_C,
        max_iter    = config.SVM_MAX_ITER,
        class_weight= 'balanced',
        random_state= config.RANDOM_SEED,
    )
    # Calibrate to get probabilities via Platt scaling
    return CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')


def build_knn(config) -> KNeighborsClassifier:
    """
    K-Nearest Neighbors with cosine distance.
    Works okay on TF-IDF vectors; very slow for large datasets.
    NOTE: For inference, KNN checks all training points — not great for prod.
    """
    return KNeighborsClassifier(
        n_neighbors = config.KNN_K,
        metric      = config.KNN_METRIC,
        weights     = config.KNN_WEIGHTS,
        algorithm   = 'brute',       # Required for cosine metric
        n_jobs      = config.N_JOBS,
    )


def build_xgboost(config):
    """
    XGBoost gradient boosting — excellent on sparse TF-IDF features.
    Uses histogram-based algorithm for speed on high-dimensional data.
    """
    if XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators        = config.XGB_N_ESTIMATORS,
            max_depth           = config.XGB_MAX_DEPTH,
            learning_rate       = config.XGB_LEARNING_RATE,
            subsample           = config.XGB_SUBSAMPLE,
            colsample_bytree    = config.XGB_COL_SAMPLE,
            use_label_encoder   = False,
            eval_metric         = 'mlogloss',
            random_state        = config.RANDOM_SEED,
            tree_method         = 'hist',
            n_jobs              = config.N_JOBS,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators    = config.XGB_N_ESTIMATORS,
            max_depth       = config.XGB_MAX_DEPTH,
            learning_rate   = config.XGB_LEARNING_RATE,
            subsample       = config.XGB_SUBSAMPLE,
            random_state    = config.RANDOM_SEED,
        )


def build_voting_ensemble(lr, svm, xgb, weights: List[int]) -> VotingClassifier:
    """
    Soft voting ensemble over Logistic Regression, SVM, and XGBoost.
    Soft voting averages class probabilities → more stable than hard voting.

    Why LR + SVM + XGB?
      • LR captures linear feature relationships
      • SVM maximizes margin, robust to outliers
      • XGB captures non-linear interactions
      • They make different types of errors → complementary
    """
    return VotingClassifier(
        estimators=[
            ('lr', lr),
            ('svm', svm),
            ('xgb', xgb),
        ],
        voting='soft',
        weights=weights,
        n_jobs=1,   # Each sub-model already parallelizes
    )


# =============================================================================
# MODEL TRAINER
# =============================================================================
class MLModelTrainer:
    """
    Trains, evaluates, and saves all 5 ML models + ensemble.
    """

    def __init__(self, config=None):
        if config is None:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from config import MLConfig
            config = MLConfig

        self.cfg = config
        self.models: Dict = {}
        self.label_encoder = LabelEncoder()

    def build_all(self) -> Dict:
        """Build all model instances."""
        print('[ML] Building models...')
        self.models = {
            'lr'  : build_logistic_regression(self.cfg),
            'rf'  : build_random_forest(self.cfg),
            'svm' : build_svm(self.cfg),
            'knn' : build_knn(self.cfg),
            'xgb' : build_xgboost(self.cfg),
        }
        print(f'[ML] Built: {list(self.models.keys())}')
        return self.models

    def train_model(
        self,
        name: str,
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
    ):
        """Train a single model and optionally evaluate on validation set."""
        print(f'\n[ML] Training {name.upper()}...')
        model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            f1  = f1_score(y_val, preds, average='weighted')
            print(f'[ML] {name.upper()} Val Acc: {acc:.4f} | Val F1: {f1:.4f}')
        return model

    def train_all(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train all 5 individual models."""
        if not self.models:
            self.build_all()

        results = {}
        for name, model in self.models.items():
            trained = self.train_model(name, model, X_train, y_train, X_val, y_val)
            self.models[name] = trained

            # Individual evaluation
            val_preds = trained.predict(X_val)
            results[name] = {
                'val_accuracy': accuracy_score(y_val, val_preds),
                'val_f1'      : f1_score(y_val, val_preds, average='weighted'),
            }

        # Build and train ensemble (LR + SVM + XGB)
        print('\n[ML] Building and training VotingEnsemble (LR + SVM + XGB)...')
        ensemble = build_voting_ensemble(
            self.models['lr'],
            self.models['svm'],
            self.models['xgb'],
            weights=self.cfg.ENSEMBLE_WEIGHTS,
        )
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble

        ens_preds = ensemble.predict(X_val)
        results['ensemble'] = {
            'val_accuracy': accuracy_score(y_val, ens_preds),
            'val_f1'      : f1_score(y_val, ens_preds, average='weighted'),
        }

        # Print summary
        print('\n' + '='*60)
        print('MODEL COMPARISON (Validation Set):')
        print('='*60)
        print(f'{"Model":<12} {"Accuracy":>10} {"F1 (weighted)":>15}')
        print('-'*40)
        for name, metrics in sorted(results.items(), key=lambda x: -x[1]['val_accuracy']):
            print(f'{name:<12} {metrics["val_accuracy"]:>10.4f} {metrics["val_f1"]:>15.4f}')
        print('='*60)

        return results

    def evaluate_on_test(self, X_test, y_test, model_name: str = 'ensemble'):
        """Full evaluation on held-out test set."""
        model = self.models[model_name]
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='weighted')
        cm  = confusion_matrix(y_test, preds)

        print(f'\n[ML] {model_name.upper()} TEST RESULTS:')
        print(f'  Accuracy : {acc:.4f}')
        print(f'  F1       : {f1:.4f}')
        print(f'\nClassification Report:')
        print(classification_report(
            y_test, preds,
            target_names=self.label_encoder.classes_,
            digits=4,
        ))
        print(f'Confusion Matrix:\n{cm}')

        return {'accuracy': acc, 'f1': f1, 'confusion_matrix': cm}

    def save_all(self, save_dir: str):
        """Save all trained models to disk."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(save_dir, f'ml_{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f'[ML] Saved {name} → {path}')

        # Also save the ensemble separately for easy loading
        ens_path = os.path.join(save_dir, 'ml_ensemble.pkl')
        with open(ens_path, 'wb') as f:
            pickle.dump(self.models.get('ensemble'), f)

    @staticmethod
    def load_model(path: str):
        """Load a single saved model."""
        with open(path, 'rb') as f:
            return pickle.load(f)
