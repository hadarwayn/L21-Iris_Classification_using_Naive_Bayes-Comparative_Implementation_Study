"""
Path management utilities for the Iris Naive Bayes project.

This module provides centralized path management for all project directories and files.
"""

from pathlib import Path
from typing import Optional


class ProjectPaths:
    """Centralized path management for the project."""

    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize project paths.

        Args:
            root_dir: Root directory of the project. If None, uses current file's parent.
        """
        if root_dir is None:
            # Get project root (3 levels up from this file)
            self.root = Path(__file__).resolve().parent.parent.parent
        else:
            self.root = Path(root_dir).resolve()

        # Main directories
        self.src = self.root / "src"
        self.data = self.root / "data"
        self.config = self.root / "config"
        self.logs = self.root / "logs"
        self.results = self.root / "results"
        self.tests = self.root / "tests"

        # Data files
        self.iris_csv = self.data / "Iris.csv"

        # Config files
        self.settings = self.config / "settings.yaml"
        self.log_config = self.logs / "config" / "log_config.json"
        self.experiments = self.config / "experiments"

        # Results subdirectories
        self.results_graphs = self.results / "graphs"
        self.results_tables = self.results / "tables"
        self.results_examples = self.results / "examples"

    def get_experiment_config(self, experiment_name: str) -> Path:
        """
        Get path to experiment configuration file.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment config file
        """
        return self.experiments / f"{experiment_name}.yaml"

    def get_run_dir(self, run_id: int) -> Path:
        """
        Get path to a specific run directory.

        Args:
            run_id: Run identifier

        Returns:
            Path to the run directory
        """
        return self.results_examples / f"run_{run_id}"

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data,
            self.config,
            self.logs,
            self.logs / "config",
            self.results,
            self.results_graphs,
            self.results_tables,
            self.results_examples,
            self.tests,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """String representation of ProjectPaths."""
        return f"ProjectPaths(root={self.root})"


# Global instance for easy access
paths = ProjectPaths()
