"""Evaluation and comparison for Naive Bayes implementations."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from typing import Dict, Any
import logging

from src.utils.helpers import benchmark_function, format_time

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates and compares Naive Bayes classifier implementations."""

    def __init__(self, average: str = 'weighted'):
        """Initialize evaluator with averaging method for multiclass metrics."""
        self.average = average

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics (accuracy, precision, recall, F1)."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=self.average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=self.average, zero_division=0)
        }

        logger.info(f"Metrics calculated: {metrics}")
        return metrics

    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion matrix shape: {cm.shape}")
        return cm

    def compare_predictions(self, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, Any]:
        """Compare predictions from two models, returning agreement statistics."""
        agreement = np.sum(y_pred1 == y_pred2)
        total = len(y_pred1)
        agreement_rate = agreement / total

        disagreement_indices = np.where(y_pred1 != y_pred2)[0]

        comparison = {
            'agreement_count': int(agreement),
            'disagreement_count': int(total - agreement),
            'agreement_rate': float(agreement_rate),
            'total_predictions': int(total),
            'disagreement_indices': disagreement_indices.tolist()
        }

        logger.info(f"Prediction agreement: {agreement}/{total} ({agreement_rate:.2%})")
        return comparison

    def compare_parameters(self, params1: Dict, params2: Dict) -> Dict[str, Any]:
        """Compare learned parameters (priors, means, variances) from two models."""
        comparison = {}

        # Compare priors
        prior_diff = np.abs(params1['class_prior'] - params2['class_prior'])
        comparison['prior_max_diff'] = float(np.max(prior_diff))
        comparison['prior_mean_diff'] = float(np.mean(prior_diff))

        # Compare means (theta)
        theta_diff = np.abs(params1['theta'] - params2['theta'])
        comparison['theta_max_diff'] = float(np.max(theta_diff))
        comparison['theta_mean_diff'] = float(np.mean(theta_diff))

        # Compare variances
        var_diff = np.abs(params1['var'] - params2['var'])
        comparison['var_max_diff'] = float(np.max(var_diff))
        comparison['var_mean_diff'] = float(np.mean(var_diff))

        logger.info(f"Parameter comparison: max prior diff={comparison['prior_max_diff']:.6f}")
        return comparison

    def benchmark_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, n_runs: int = 10) -> Dict[str, float]:
        """Benchmark model training and prediction time over multiple runs."""
        # Benchmark training
        _, train_mean, train_std = benchmark_function(model.fit, n_runs, X_train, y_train)

        # Benchmark prediction
        _, pred_mean, pred_std = benchmark_function(model.predict, n_runs, X_test)

        timing = {
            'train_time_mean': train_mean,
            'train_time_std': train_std,
            'predict_time_mean': pred_mean,
            'predict_time_std': pred_std,
            'train_time_formatted': format_time(train_mean),
            'predict_time_formatted': format_time(pred_mean)
        }

        logger.info(f"Training time: {timing['train_time_formatted']} ± {format_time(train_std)}")
        logger.info(f"Prediction time: {timing['predict_time_formatted']} ± {format_time(pred_std)}")

        return timing
