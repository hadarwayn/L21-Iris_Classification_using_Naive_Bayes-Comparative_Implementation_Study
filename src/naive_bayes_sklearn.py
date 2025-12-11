"""
Scikit-learn wrapper for Gaussian Naive Bayes classifier.

This module provides a thin wrapper around scikit-learn's GaussianNB
for comparison with the manual NumPy implementation.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from typing import Optional
import logging

from src.utils.validators import validate_numpy_array

logger = logging.getLogger(__name__)


class GaussianNaiveBayesSklearn:
    """
    Wrapper around scikit-learn's Gaussian Naive Bayes classifier.

    Provides consistent interface with the manual NumPy implementation.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize sklearn Gaussian Naive Bayes wrapper.

        Args:
            var_smoothing: Portion of largest variance added to variances for stability
        """
        self.var_smoothing = var_smoothing
        self.model = GaussianNB(var_smoothing=var_smoothing)
        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None  # Means
        self.var_: Optional[np.ndarray] = None  # Variances (before smoothing)
        self.n_classes_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayesSklearn':
        """
        Fit Gaussian Naive Bayes classifier using scikit-learn.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Returns:
            Self for method chaining
        """
        # Validate inputs
        validate_numpy_array(X, ndim=2, allow_nan=False)
        validate_numpy_array(y, ndim=1, allow_nan=False)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y size mismatch: {X.shape[0]} vs {y.shape[0]}")

        # Fit the model
        self.model.fit(X, y)

        # Extract learned parameters for comparison
        self.classes_ = self.model.classes_
        self.class_prior_ = self.model.class_prior_
        self.theta_ = self.model.theta_

        # sklearn stores sigma_ (variance after smoothing), we want raw variance
        # sigma_ = var_ + var_smoothing * np.var(X, axis=0).max()
        # To get original variance, we reverse this
        max_var = np.var(X, axis=0).max()
        self.var_ = self.model.var_ - self.var_smoothing * max_var

        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        logger.info(f"Fitted sklearn model on {X.shape[0]} samples with {self.n_features_} features")
        logger.info(f"Classes: {self.classes_}")
        logger.info(f"Class priors: {self.class_prior_}")

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate log posterior probabilities for each class.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Log probabilities of shape (n_samples, n_classes)
        """
        if self.classes_ is None:
            raise ValueError("Model not fitted. Call fit() before predict.")

        validate_numpy_array(X, ndim=2, allow_nan=False)

        return self.model.predict_log_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        if self.classes_ is None:
            raise ValueError("Model not fitted. Call fit() before predict.")

        validate_numpy_array(X, ndim=2, allow_nan=False)

        return self.model.predict(X)

    def get_params(self) -> dict:
        """
        Get learned parameters.

        Returns:
            Dictionary containing classes, priors, means, and variances
        """
        if self.classes_ is None:
            raise ValueError("Model not fitted. Call fit() before get_params.")

        return {
            'classes': self.classes_,
            'class_prior': self.class_prior_,
            'theta': self.theta_,
            'var': self.var_,
            'n_classes': self.n_classes_,
            'n_features': self.n_features_
        }
