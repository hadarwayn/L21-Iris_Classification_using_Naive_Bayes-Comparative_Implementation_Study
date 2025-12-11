"""
Input validation utilities for the Iris Naive Bayes project.

This module provides validation functions for data integrity checks.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> None:
    """
    Validate that a DataFrame meets basic requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValueError: If validation fails
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # Check for missing values
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"DataFrame contains missing values in columns: {null_cols}")


def validate_numpy_array(X: np.ndarray, ndim: int = 2, allow_nan: bool = False) -> None:
    """
    Validate NumPy array properties.

    Args:
        X: Array to validate
        ndim: Expected number of dimensions
        allow_nan: Whether to allow NaN values

    Raises:
        ValueError: If validation fails
        TypeError: If input is not a NumPy array
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(X)}")

    if X.ndim != ndim:
        raise ValueError(f"Expected {ndim}D array, got {X.ndim}D array with shape {X.shape}")

    if X.size == 0:
        raise ValueError("Array is empty")

    # Only check for NaN/Inf on numeric arrays
    if np.issubdtype(X.dtype, np.number):
        if not allow_nan and np.isnan(X).any():
            raise ValueError("Array contains NaN values")

        if np.isinf(X).any():
            raise ValueError("Array contains infinite values")


def validate_train_test_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    Validate train-test split data.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels

    Raises:
        ValueError: If validation fails
    """
    # Validate shapes
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and y_train size mismatch: {X_train.shape[0]} vs {y_train.shape[0]}"
        )

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"X_test and y_test size mismatch: {X_test.shape[0]} vs {y_test.shape[0]}"
        )

    # Validate feature dimensions match
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}"
        )

    # Validate arrays
    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        validate_numpy_array(arr, ndim=2, allow_nan=False)

    for name, arr in [("y_train", y_train), ("y_test", y_test)]:
        validate_numpy_array(arr, ndim=1, allow_nan=False)


def validate_class_distribution(y: np.ndarray, min_samples_per_class: int = 2) -> None:
    """
    Validate that each class has minimum required samples.

    Args:
        y: Label array
        min_samples_per_class: Minimum samples required per class

    Raises:
        ValueError: If any class has too few samples
    """
    unique, counts = np.unique(y, return_counts=True)

    insufficient = [(cls, cnt) for cls, cnt in zip(unique, counts) if cnt < min_samples_per_class]

    if insufficient:
        raise ValueError(
            f"Classes with insufficient samples (min={min_samples_per_class}): {insufficient}"
        )


def validate_probability_array(probs: np.ndarray, sum_to_one: bool = True, tolerance: float = 1e-6) -> None:
    """
    Validate probability array.

    Args:
        probs: Probability array
        sum_to_one: Whether probabilities should sum to 1
        tolerance: Tolerance for sum check

    Raises:
        ValueError: If validation fails
    """
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be in [0, 1]")

    if sum_to_one:
        prob_sum = np.sum(probs)
        if not np.isclose(prob_sum, 1.0, atol=tolerance):
            raise ValueError(f"Probabilities sum to {prob_sum}, expected 1.0")
