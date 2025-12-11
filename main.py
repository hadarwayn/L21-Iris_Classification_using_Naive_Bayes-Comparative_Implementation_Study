"""Main orchestration script for Iris Naive Bayes comparison."""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import logging

from src.utils.paths import paths
from src.utils.helpers import load_yaml_config, save_yaml_config
from src.utils.logger import setup_logger
from src.data_loader import IrisDataLoader
from src.naive_bayes_manual import GaussianNaiveBayesManual
from src.naive_bayes_sklearn import GaussianNaiveBayesSklearn
from src.evaluator import ModelEvaluator
from src.visualizer import ResultsVisualizer

def run_experiment(config_path: Path) -> Dict[str, Any]:
    """Run complete experiment: load data, train models, evaluate, visualize."""
    config = load_yaml_config(config_path)
    exp_name = config['experiment']['name']
    logger = setup_logger(f"experiment_{exp_name}", paths.logs, logging.INFO, 10000, 5)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Configuration: {config}")
    output_dir = Path(config['output']['results_subdir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading data...")
    data_loader = IrisDataLoader(data_path=paths.iris_csv)
    data_loader.load()
    X_train, X_test, y_train, y_test = data_loader.split_data(
        config['data']['test_size'], config['data']['random_seed'],
        config['data']['stratify'], config['data']['normalize']
    )
    classes = sorted(data_loader.df[data_loader.TARGET_COLUMN].unique())
    logger.info("Initializing and training models...")
    model_numpy = GaussianNaiveBayesManual(config['models']['numpy']['epsilon'])
    model_sklearn = GaussianNaiveBayesSklearn(config['models']['sklearn']['var_smoothing'])
    model_numpy.fit(X_train, y_train)
    model_sklearn.fit(X_train, y_train)
    logger.info("Making predictions...")
    y_pred_numpy = model_numpy.predict(X_test)
    y_pred_sklearn = model_sklearn.predict(X_test)
    logger.info("Evaluating models...")
    evaluator = ModelEvaluator('weighted')
    metrics_numpy = evaluator.calculate_metrics(y_test, y_pred_numpy)
    metrics_sklearn = evaluator.calculate_metrics(y_test, y_pred_sklearn)
    cm_numpy = evaluator.get_confusion_matrix(y_test, y_pred_numpy)
    cm_sklearn = evaluator.get_confusion_matrix(y_test, y_pred_sklearn)
    pred_comparison = evaluator.compare_predictions(y_pred_numpy, y_pred_sklearn)
    params_numpy = {'class_prior': model_numpy.class_prior_, 'theta': model_numpy.theta_, 'var': model_numpy.var_}
    params_sklearn = model_sklearn.get_params()
    param_comparison = evaluator.compare_parameters(params_numpy, params_sklearn)
    logger.info("Benchmarking runtime...")
    timing_numpy = evaluator.benchmark_model(
        GaussianNaiveBayesManual(config['models']['numpy']['epsilon']),
        X_train, y_train, X_test, config['evaluation']['benchmark_runs']
    )
    timing_sklearn = evaluator.benchmark_model(
        GaussianNaiveBayesSklearn(config['models']['sklearn']['var_smoothing']),
        X_train, y_train, X_test, config['evaluation']['benchmark_runs']
    )

    logger.info("Creating visualizations...")
    visualizer = ResultsVisualizer(output_dir / 'plots', 300)
    visualizer.plot_confusion_matrix(cm_numpy, classes, 'Confusion Matrix - NumPy', 'cm_numpy.png')
    visualizer.plot_confusion_matrix(cm_sklearn, classes, 'Confusion Matrix - Sklearn', 'cm_sklearn.png')
    visualizer.plot_metrics_comparison({'NumPy': metrics_numpy, 'Sklearn': metrics_sklearn}, 'metrics_comparison.png')
    visualizer.plot_runtime_comparison({'NumPy': timing_numpy, 'Sklearn': timing_sklearn}, 'runtime_comparison.png')
    visualizer.plot_feature_distributions(X_train, y_train, data_loader.FEATURE_COLUMNS, classes, 'feature_distributions.png')
    visualizer.plot_class_distribution(y_train, classes, 'class_distribution.png')
    results = {
        'experiment': config['experiment'],
        'metrics': {'numpy': metrics_numpy, 'sklearn': metrics_sklearn},
        'timing': {'numpy': timing_numpy, 'sklearn': timing_sklearn},
        'prediction_comparison': pred_comparison,
        'parameter_comparison': param_comparison,
        'data_info': {'train_size': len(X_train), 'test_size': len(X_test), 'n_features': X_train.shape[1], 'classes': classes}
    }
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")
    logger.info("Experiment completed successfully!")
    return results


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment_config.yaml>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    results = run_experiment(config_path)
    print("\n=== Experiment Complete ===")
    print(f"NumPy Accuracy: {results['metrics']['numpy']['accuracy']:.4f}")
    print(f"Sklearn Accuracy: {results['metrics']['sklearn']['accuracy']:.4f}")
    print(f"Prediction Agreement: {results['prediction_comparison']['agreement_rate']:.2%}")

if __name__ == "__main__":
    main()
