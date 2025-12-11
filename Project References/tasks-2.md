# Implementation Tasks

This document tracks all tasks required to implement the Naive Bayes Analysis & Comparison Tool as specified in PRD.md.

## Legend
- [x] Not started
- [~] In progress
- [x] Completed

---

## Phase 1: Project Setup & Configuration

### Environment Setup
- [x] Create `pyproject.toml` with UV configuration
  - [x] Define project metadata (name, version, description)
  - [x] Specify Python version requirement (>=3.8)
  - [x] List dependencies: numpy, scikit-learn, matplotlib, pandas
  - [x] Configure build system

- [x] Initialize UV environment
  - [x] Run `uv init` if needed
  - [x] Run `uv sync` to create lock file
  - [x] Verify environment works

### Package Structure
- [x] Create `src/__init__.py` to make src a proper package
- [x] Create `analysis/` directory
- [x] Create `analysis/__init__.py`
- [x] Create `output/` directory
- [x] Create `output/plots/` directory
- [x] Add `.gitignore` for Python projects

---

## Phase 2: Source Code Refactoring

### Prepare Existing Code
- [x] Refactor `src/naive_bayes_numpy.py`
  - [x] Extract training logic into reusable function
  - [x] Extract prediction logic into reusable function
  - [x] Return model components (means, stds, priors, classes)
  - [x] Accept train/test data as parameters

- [x] Refactor `src/naive_bayes_sklearn.py`
  - [x] Extract training logic into reusable function
  - [x] Extract prediction logic into reusable function
  - [x] Return trained model
  - [x] Accept train/test data as parameters

- [x] Ensure both use same data split
  - [x] Centralize dataset loading
  - [x] Use fixed random_state=42
  - [x] Use stratified split

---

## Phase 3: Core Comparison Module

### `analysis/comparison.py` (≤200 lines)

- [x] Import and setup
  - [x] Import numpy, sklearn modules
  - [x] Import from src.naive_bayes_numpy and src.naive_bayes_sklearn

- [x] Data loading function
  - [x] `load_dataset()` - Load Iris, split train/test
  - [x] Return X_train, X_test, y_train, y_test

- [x] Model execution functions
  - [x] `run_numpy_model(X_train, y_train, X_test)` - Train and predict with NumPy
  - [x] `run_sklearn_model(X_train, y_train, X_test)` - Train and predict with sklearn
  - [x] Both return predictions and model objects

- [x] Accuracy metrics
  - [x] `calculate_accuracy(y_true, y_pred)` - Overall accuracy
  - [x] `calculate_per_class_metrics(y_true, y_pred)` - Precision, recall, F1
  - [x] `calculate_confusion_matrix(y_true, y_pred)` - Confusion matrix

- [x] Error metrics
  - [x] `calculate_mse(y_true, y_pred)` - Mean Squared Error
  - [x] `calculate_rmse(y_true, y_pred)` - Root Mean Squared Error
  - [x] `calculate_mae(y_true, y_pred)` - Mean Absolute Error

- [x] Comparison function
  - [x] `compare_predictions(y_true, y_pred_numpy, y_pred_sklearn)` - Compare both
  - [x] Return comprehensive metrics dictionary

- [x] Verify line count ≤200

---

## Phase 4: Visualization Module

### `analysis/visualizer.py` (≤200 lines)

- [x] Import and setup
  - [x] Import matplotlib.pyplot, sklearn.decomposition, sklearn.discriminant_analysis
  - [x] Configure matplotlib style/settings

- [x] PCA visualization
  - [x] `plot_pca(X_test, y_true, y_pred, title, save_path)` - Generic PCA plotter
  - [x] Reduce to 2 components
  - [x] Plot correct predictions as filled circles
  - [x] Plot incorrect predictions as X markers
  - [x] Color by true labels
  - [x] Add legend and labels
  - [x] Save to file

- [x] LDA visualization
  - [x] `plot_lda(X_test, y_true, y_pred, title, save_path)` - Generic LDA plotter
  - [x] Reduce to 2 components
  - [x] Plot correct predictions as filled circles
  - [x] Plot incorrect predictions as triangles
  - [x] Color by true labels
  - [x] Add legend and labels
  - [x] Save to file

- [x] Comparison visualization
  - [x] `plot_comparison(X_test, y_true, y_pred_numpy, y_pred_sklearn, save_dir)` - Combined plot
  - [x] Create 2x2 subplot: PCA NumPy, PCA sklearn, LDA NumPy, LDA sklearn
  - [x] Save combined figure

- [x] Helper functions
  - [x] `get_misclassified_indices(y_true, y_pred)` - Find errors
  - [x] `create_plot_title(method, implementation)` - Format titles

- [x] Main visualization function
  - [x] `visualize_all(X_test, y_true, y_pred_numpy, y_pred_sklearn, output_dir)`
  - [x] Generate all plots
  - [x] Return list of generated files

- [x] Verify line count ≤200

---

## Phase 5: Benchmarking Module

### `analysis/benchmarker.py` (≤200 lines)

- [x] Import and setup
  - [x] Import time, statistics modules
  - [x] Import model functions from comparison

- [x] Timing utilities
  - [x] `time_function(func, *args, **kwargs)` - Measure single execution
  - [x] Return elapsed time and result

- [x] Benchmark functions
  - [x] `benchmark_numpy(X_train, y_train, X_test, n_runs=5)` - Time NumPy impl
    - [x] Measure training time
    - [x] Measure inference time
    - [x] Run multiple times
    - [x] Return mean, std, all times

  - [x] `benchmark_sklearn(X_train, y_train, X_test, n_runs=5)` - Time sklearn impl
    - [x] Measure training time
    - [x] Measure inference time
    - [x] Run multiple times
    - [x] Return mean, std, all times

- [x] Statistics calculations
  - [x] `calculate_statistics(times)` - Mean, std, min, max
  - [x] `calculate_speedup(time1, time2)` - Speedup factor
  - [x] `calculate_percentage_diff(time1, time2)` - Percentage difference

- [x] Main benchmark function
  - [x] `run_benchmarks(X_train, y_train, X_test, n_runs=5)`
  - [x] Benchmark both implementations
  - [x] Calculate comparisons
  - [x] Return comprehensive timing dictionary

- [x] Verify line count ≤200

---

## Phase 6: Reporting Module

### `analysis/reporter.py` (≤200 lines)

- [x] Import and setup
  - [x] Import datetime for timestamps
  - [x] Import pathlib for file operations

- [x] Formatting utilities
  - [x] `format_float(value, decimals=4)` - Format numbers
  - [x] `format_table(headers, rows)` - Create markdown table
  - [x] `format_percentage(value)` - Format as percentage

- [x] Report sections
  - [x] `generate_header(timestamp)` - Report title and timestamp
  - [x] `generate_dataset_info(X_train, X_test, y_train, y_test)` - Dataset summary
  - [x] `generate_accuracy_section(metrics_numpy, metrics_sklearn)` - Accuracy tables
  - [x] `generate_error_section(errors_numpy, errors_sklearn)` - Error metrics
  - [x] `generate_runtime_section(benchmark_results)` - Timing tables
  - [x] `generate_visual_section(plot_files)` - Links to plots
  - [x] `generate_conclusions(metrics_numpy, metrics_sklearn, benchmarks)` - Summary

- [x] Main report function
  - [x] `generate_report(metrics, benchmarks, plot_files, output_path)`
  - [x] Combine all sections
  - [x] Write to markdown file
  - [x] Return success status

- [x] Markdown generation
  - [x] Create proper heading hierarchy
  - [x] Format tables correctly
  - [x] Add plot references
  - [x] Include statistics

- [x] Verify line count ≤200

---

## Phase 7: Main Entry Point

### `analysis/__main__.py` (≤200 lines)

- [x] Import all modules
  - [x] From comparison import comparison functions
  - [x] From visualizer import visualization functions
  - [x] From benchmarker import benchmark functions
  - [x] From reporter import report generation

- [x] Main orchestration
  - [x] `main()` function
    - [x] Print welcome message
    - [x] Load dataset
    - [x] Run both models
    - [x] Calculate metrics
    - [x] Run benchmarks
    - [x] Generate visualizations
    - [x] Generate report
    - [x] Print summary

- [x] Error handling
  - [x] Wrap in try-except
  - [x] Handle missing dependencies
  - [x] Handle file I/O errors
  - [x] Provide helpful error messages

- [x] Output management
  - [x] Create output directories if needed
  - [x] Check file permissions
  - [x] Log progress to console

- [x] Entry point
  - [x] `if __name__ == "__main__":` block
  - [x] Call main()

- [x] Verify line count ≤200

---

## Phase 8: Testing & Validation

### Code Quality
- [x] Line count verification
  - [x] comparison.py ≤200 lines
  - [x] visualizer.py ≤200 lines
  - [x] benchmarker.py ≤200 lines
  - [x] reporter.py ≤200 lines
  - [x] __main__.py ≤200 lines

### Functional Testing
- [x] Test on clean environment
  - [x] Create fresh UV environment
  - [x] Run `uv sync`
  - [x] Execute `uv run python -m analysis`

- [x] Verify outputs
  - [x] Check all plots generated in `output/plots/`
  - [x] Check `docs/analysis_report.md` created
  - [x] Verify report content is accurate
  - [x] Validate plot quality

- [x] Cross-platform testing (if possible)
  - [x] Test on Windows
  - [x] Test on Linux/macOS

### Metrics Validation
- [x] Manual verification
  - [x] Calculate accuracy manually, compare with tool output
  - [x] Verify MSE calculation
  - [x] Check confusion matrix
  - [x] Validate timing makes sense

---

## Phase 9: Documentation

### Code Documentation
- [x] Add docstrings to all functions
  - [x] comparison.py functions
  - [x] visualizer.py functions
  - [x] benchmarker.py functions
  - [x] reporter.py functions
  - [x] __main__.py main function

- [x] Add inline comments for complex logic
  - [x] PCA/LDA dimensionality reduction
  - [x] Error metric calculations
  - [x] Timing measurement

### Project Documentation
- [x] Update `CLAUDE.md`
  - [x] Add analysis module commands
  - [x] Document new directory structure
  - [x] Add usage examples
  - [x] Document output files

- [x] Create usage examples
  - [x] Basic usage in README (if created)
  - [x] Command reference

---

## Phase 10: Final Deliverables

### Code Deliverables
- [x] `pyproject.toml` - UV configuration
- [x] `uv.lock` - Locked dependencies
- [x] `src/__init__.py` - Make src a package
- [x] `analysis/__init__.py` - Package initialization
- [x] `analysis/comparison.py` - Comparison logic (≤200 lines)
- [x] `analysis/visualizer.py` - Visualization (≤200 lines)
- [x] `analysis/benchmarker.py` - Benchmarking (≤200 lines)
- [x] `analysis/reporter.py` - Reporting (≤200 lines)
- [x] `analysis/__main__.py` - Entry point (≤200 lines)

### Output Deliverables
- [x] `output/plots/pca_numpy.png`
- [x] `output/plots/pca_sklearn.png`
- [x] `output/plots/lda_numpy.png`
- [x] `output/plots/lda_sklearn.png`
- [x] `output/plots/comparison.png` (combined)
- [x] `docs/analysis_report.md`

### Documentation Deliverables
- [x] Updated `CLAUDE.md`
- [x] Inline code documentation (docstrings)
- [x] This TASKS.md (updated with completion status)

---

## Summary

**Total Tasks**: ~120+ individual tasks
**Estimated Time**: 4-6 hours for full implementation
**Critical Path**: Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7

**Dependencies**:
- Phase 3 depends on Phase 2 (refactored source code)
- Phases 4, 5, 6 depend on Phase 3 (comparison functions)
- Phase 7 depends on Phases 3, 4, 5, 6 (all modules complete)
- Phase 8 depends on Phase 7 (everything implemented)
- Phase 9 runs parallel to other phases
- Phase 10 is final verification

**Success Criteria**:
- All modules ≤200 lines
- All outputs generated correctly
- Report is accurate and comprehensive
- Single command execution works: `uv run python -m analysis`
