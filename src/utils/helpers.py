"""General helper functions for the Iris Naive Bayes project."""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import time
from functools import wraps

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def save_yaml_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration dictionary to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)


def normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.

    Args:
        X: Feature array of shape (n_samples, n_features)

    Returns:
        Tuple of (normalized_X, means, stds)
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    # Avoid division by zero
    stds = np.where(stds == 0, 1, stds)

    X_normalized = (X - means) / stds

    return X_normalized, means, stds


def timer(func):
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.6f} seconds")
        return result
    return wrapper


def benchmark_function(func, n_runs: int = 10, *args, **kwargs) -> tuple[Any, float, float]:
    """
    Benchmark a function by running it multiple times.

    Args:
        func: Function to benchmark
        n_runs: Number of times to run the function
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (result, mean_time, std_time)
    """
    times = []
    result = None

    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return result, mean_time, std_time


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.4f} s"


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.

    Returns:
        Timestamp string in YYYYMMDD_HHMMSS format
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")
