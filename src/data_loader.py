"""Data loading and preprocessing for Iris Naive Bayes project."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import logging

from src.utils.validators import (
    validate_dataframe,
    validate_train_test_split,
    validate_class_distribution
)
from src.utils.helpers import normalize_features, set_random_seeds

logger = logging.getLogger(__name__)

class IrisDataLoader:
    """Handles loading and preprocessing of the Iris dataset."""

    REQUIRED_COLUMNS = [
        'SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm',
        'Species'
    ]

    FEATURE_COLUMNS = [
        'SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm'
    ]

    TARGET_COLUMN = 'Species'

    def __init__(self, data_path: Path):
        """
        Initialize data loader.

        Args:
            data_path: Path to Iris.csv file
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.normalization_params: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def load(self) -> pd.DataFrame:
        """
        Load the Iris dataset from CSV.

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # Validate DataFrame
        validate_dataframe(self.df, required_columns=self.REQUIRED_COLUMNS)

        logger.info(f"Loaded {len(self.df)} samples with {len(self.FEATURE_COLUMNS)} features")
        logger.info(f"Class distribution:\n{self.df[self.TARGET_COLUMN].value_counts()}")

        return self.df

    def get_features_and_labels(self, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features (X) and labels (y) from the dataset.

        Args:
            normalize: Whether to normalize features using z-score

        Returns:
            Tuple of (X, y) where X is features array and y is labels array
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        X = self.df[self.FEATURE_COLUMNS].values
        y = self.df[self.TARGET_COLUMN].values

        if normalize:
            X, means, stds = normalize_features(X)
            self.normalization_params = (means, stds)
            logger.info("Features normalized using z-score normalization")

        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")

        return X, y

    def split_data(
        self, test_size: float = 0.2, random_seed: int = 42, stratify: bool = True,
        normalize: bool = False, min_samples_per_class: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets with optional normalization and stratification."""
        # Set random seeds
        set_random_seeds(random_seed)

        # Get features and labels
        X, y = self.get_features_and_labels(normalize=normalize)

        # Validate class distribution
        validate_class_distribution(y, min_samples_per_class=min_samples_per_class)

        # Perform train-test split
        stratify_arg = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_arg
        )

        # Validate split
        validate_train_test_split(X_train, X_test, y_train, y_test)

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train class distribution: {np.unique(y_train, return_counts=True)}")
        logger.info(f"Test class distribution: {np.unique(y_test, return_counts=True)}")

        return X_train, X_test, y_train, y_test
