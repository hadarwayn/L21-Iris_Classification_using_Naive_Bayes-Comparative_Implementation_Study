"""Manual Gaussian Naive Bayes implementation using only NumPy."""

import numpy as np
from typing import Optional
import logging

from src.utils.validators import validate_numpy_array

logger = logging.getLogger(__name__)

class GaussianNaiveBayesManual:
    """Gaussian Naive Bayes classifier using Bayes' theorem with Gaussian PDF assumption."""

    def __init__(self, epsilon: float = 1e-9):
        """Initialize classifier with epsilon for numerical stability."""
        self.epsilon = epsilon
        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None  # Means
        self.var_: Optional[np.ndarray] = None  # Variances
        self.n_classes_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayesManual':
        """
        Fit Gaussian Naive Bayes classifier.

        Learns class priors, feature means, and variances for each class.

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

        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Initialize parameter arrays
        self.class_prior_ = np.zeros(self.n_classes_)
        self.theta_ = np.zeros((self.n_classes_, self.n_features_))
        self.var_ = np.zeros((self.n_classes_, self.n_features_))

        # Calculate parameters for each class
        for idx, class_label in enumerate(self.classes_):
            # Get samples for this class
            X_class = X[y == class_label]

            # Calculate prior: P(class) = count(class) / total_samples
            self.class_prior_[idx] = X_class.shape[0] / X.shape[0]

            # Calculate mean: μ = average of feature values for this class
            self.theta_[idx] = np.mean(X_class, axis=0)

            # Calculate variance: σ² = variance of feature values + epsilon
            self.var_[idx] = np.var(X_class, axis=0) + self.epsilon

        logger.info(f"Fitted on {X.shape[0]} samples with {self.n_features_} features")
        logger.info(f"Classes: {self.classes_}")
        logger.info(f"Class priors: {self.class_prior_}")

        return self

    def _calculate_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate log likelihood for each class.

        Uses Gaussian PDF: P(x|class) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        Log form: log P(x|class) = -0.5 * (log(2πσ²) + ((x-μ)²/σ²))

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Log likelihoods of shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        log_likelihood = np.zeros((n_samples, self.n_classes_))

        for idx in range(self.n_classes_):
            # Get parameters for this class
            mean = self.theta_[idx]
            var = self.var_[idx]

            # Calculate log likelihood for Gaussian distribution
            # log P(x|class) = -0.5 * sum(log(2πσ²) + ((x-μ)²/σ²))
            log_2pi_var = np.log(2 * np.pi * var)
            squared_diff = ((X - mean) ** 2) / var

            log_likelihood[:, idx] = -0.5 * np.sum(log_2pi_var + squared_diff, axis=1)

        return log_likelihood

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

        # Calculate log posterior: log P(class|x) = log P(x|class) + log P(class)
        log_prior = np.log(self.class_prior_)
        log_likelihood = self._calculate_log_likelihood(X)
        log_posterior = log_likelihood + log_prior

        return log_posterior

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        log_posterior = self.predict_log_proba(X)

        # Select class with maximum posterior probability
        class_indices = np.argmax(log_posterior, axis=1)
        predictions = self.classes_[class_indices]

        return predictions
