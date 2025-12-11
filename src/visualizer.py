"""
Visualization module for Naive Bayes comparison results.

This module creates publication-quality visualizations at 300 DPI.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Creates visualizations for model comparison results."""

    def __init__(self, output_dir: Path, dpi: int = 300, figsize: tuple = (10, 8)):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            dpi: Resolution for saved plots
            figsize: Default figure size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        sns.set_style("whitegrid")

    def plot_confusion_matrix(
        self, cm: np.ndarray, classes: List[str], title: str, filename: str
    ) -> None:
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix: {filename}")

    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]], filename: str) -> None:
        """Plot bar chart comparing metrics between models."""
        df = pd.DataFrame(metrics_dict).T
        ax = df.plot(kind='bar', figsize=self.figsize, width=0.8)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved metrics comparison: {filename}")

    def plot_runtime_comparison(self, timing_dict: Dict[str, Dict[str, float]], filename: str) -> None:
        """Plot runtime comparison between models."""
        models = list(timing_dict.keys())
        train_times = [timing_dict[m]['train_time_mean'] * 1000 for m in models]  # Convert to ms
        pred_times = [timing_dict[m]['predict_time_mean'] * 1000 for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.bar(x - width/2, train_times, width, label='Training Time')
        ax.bar(x + width/2, pred_times, width, label='Prediction Time')

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('Runtime Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved runtime comparison: {filename}")

    def plot_feature_distributions(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                                   classes: List[str], filename: str) -> None:
        """Plot feature distributions by class."""
        n_features = len(feature_names)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            for class_label in np.unique(y):
                class_idx = np.where(np.array(classes) == class_label)[0][0]
                data = X[y == class_label, idx]
                ax.hist(data, bins=20, alpha=0.6, label=class_label)

            ax.set_xlabel(feature_name, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature distributions: {filename}")

    def plot_class_distribution(self, y: np.ndarray, classes: List[str], filename: str) -> None:
        """Plot class distribution pie chart."""
        unique, counts = np.unique(y, return_counts=True)
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved class distribution: {filename}")
