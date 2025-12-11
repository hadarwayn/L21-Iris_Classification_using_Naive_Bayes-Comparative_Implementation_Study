# Product Requirements Document: Naive Bayes Analysis & Comparison Tool

## 1. Overview

Create a comprehensive analysis and comparison module for evaluating the two Naive Bayes implementations (NumPy and scikit-learn) in the Bayes project. The tool will visualize prediction differences, compare computational performance, and generate detailed statistical analysis reports.

## 2. Purpose

Enable data-driven comparison between custom (NumPy) and standard library (scikit-learn) Naive Bayes implementations through:
- Visual representation of prediction errors in 2D feature space
- Runtime performance benchmarking
- Statistical accuracy and error metrics
- Automated report generation

## 3. Functional Requirements

### 3.1 Prediction Comparison & Visualization

**FR1.1**: Load and evaluate both implementations
- Execute both Naive Bayes implementations on the same dataset
- Capture predictions from both models on the same test set
- Store true labels, NumPy predictions, and scikit-learn predictions

**FR1.2**: Dimensionality Reduction & Visualization
- Implement 2D visualization using **PCA** (Principal Component Analysis)
  - Reduce test features to 2 principal components
  - Plot test samples colored by true labels
  - Overlay predictions with different markers for misclassifications

- Implement 2D visualization using **LDA** (Linear Discriminant Analysis)
  - Reduce test features to 2 linear discriminants
  - Plot test samples colored by true labels
  - Overlay predictions with different markers for misclassifications

**FR1.3**: Visual Distinction
- Correct predictions: filled circles/dots
- Incorrect predictions (NumPy): X markers
- Incorrect predictions (sklearn): triangle markers
- Generate separate plots for NumPy and sklearn implementations
- Create combined comparison plot showing both

### 3.2 Runtime Performance Comparison

**FR2.1**: Benchmarking
- Measure execution time for both implementations:
  - Training time (fitting to training data)
  - Inference time (making predictions on test set)
  - Total time
- Run benchmarks multiple times (at least 5) and report:
  - Individual timings
  - Average time
  - Standard deviation

**FR2.2**: Performance Report
- Display timing comparison in table format
- Calculate speedup factor (sklearn vs NumPy or vice versa)
- Show percentage difference

### 3.3 Statistical Analysis Report

**FR3.1**: Accuracy Metrics
- Overall accuracy for both implementations
- Per-class accuracy (for each of 3 Iris classes)
- Precision, recall, F1-score for each class
- Confusion matrix

**FR3.2**: Error Metrics
- Mean Squared Error (MSE): `MSE = (1/n) * Σ(predicted - actual)²`
- Average Error Squared (same as MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

**FR3.3**: Detailed Report Generation
- Generate markdown report file: `docs/analysis_report.md`
- Include:
  - Executive summary of findings
  - Accuracy comparison table
  - Error metrics comparison table
  - Runtime performance table
  - Visual analysis section (reference to generated plots)
  - Conclusion and recommendations

## 4. Technical Requirements

### 4.1 Project Structure

```
Bayes/
├── src/
│   ├── __init__.py
│   ├── naive_bayes_numpy.py (existing)
│   └── naive_bayes_sklearn.py (existing)
├── analysis/
│   ├── __init__.py
│   ├── comparison.py (max 200 lines)
│   ├── visualizer.py (max 200 lines)
│   ├── benchmarker.py (max 200 lines)
│   └── reporter.py (max 200 lines)
├── output/
│   └── plots/ (generated)
├── tests/
│   └── test_analysis.py (optional)
├── docs/
│   ├── PRD.md (this file)
│   └── analysis_report.md (generated output)
├── pyproject.toml (UV configuration)
├── uv.lock (UV lock file)
└── CLAUDE.md (existing)
```

### 4.2 Dependencies

Required packages:
- `numpy >= 1.21.0`
- `scikit-learn >= 1.0.0`
- `matplotlib >= 3.5.0`
- `pandas >= 1.3.0` (optional, for report formatting)

### 4.3 Module Structure

The analysis package consists of:

- **comparison.py**: Core logic for comparing predictions (max 200 lines)
  - Load Iris dataset with same split as original implementations
  - Run both NumPy and scikit-learn implementations
  - Compute prediction differences
  - Calculate accuracy and error metrics

- **visualizer.py**: Visualization functions (max 200 lines)
  - PCA dimensionality reduction and plotting
  - LDA dimensionality reduction and plotting
  - Error visualization (misclassified samples)
  - Plot saving to `output/plots/`

- **benchmarker.py**: Performance measurement (max 200 lines)
  - Time both implementations (training + inference)
  - Run multiple iterations for statistical significance
  - Calculate mean, std deviation, speedup

- **reporter.py**: Report generation (max 200 lines)
  - Format analysis results
  - Generate markdown report
  - Create summary statistics tables

### 4.4 UV Virtual Environment

- Use `uv` as the package manager
- Configuration via `pyproject.toml`
- All dependencies locked in `uv.lock`
- Installation: `uv sync`
- Run module: `uv run python -m analysis`

### 4.5 Code Quality Constraints

- **Max 200 lines per file**: Each module must not exceed 200 lines
  - Includes imports, docstrings, and all code
  - Encourages modular, focused design
  - Forces separation of concerns

- **Module structure**: Package must be importable as `analysis`
  - Proper `__init__.py` files
  - Clear interface between modules
  - Reusable functions

## 5. Deliverables

### 5.1 Code Deliverables
- [ ] `analysis/__init__.py` - Package initialization
- [ ] `analysis/comparison.py` - Prediction comparison logic
- [ ] `analysis/visualizer.py` - Plotting functions (PCA & LDA)
- [ ] `analysis/benchmarker.py` - Performance benchmarking
- [ ] `analysis/reporter.py` - Report generation
- [ ] `analysis/__main__.py` - Entry point for running as module
- [ ] `pyproject.toml` - Project configuration for UV
- [ ] `src/__init__.py` - Make src a proper package

### 5.2 Output Deliverables
- [ ] `output/plots/pca_numpy.png` - NumPy PCA visualization
- [ ] `output/plots/pca_sklearn.png` - scikit-learn PCA visualization
- [ ] `output/plots/lda_numpy.png` - NumPy LDA visualization
- [ ] `output/plots/lda_sklearn.png` - scikit-learn LDA visualization
- [ ] `output/plots/comparison.png` - Combined comparison
- [ ] `docs/analysis_report.md` - Generated analysis report

### 5.3 Documentation
- [ ] Updated `CLAUDE.md` with analysis commands
- [ ] Inline docstrings for all functions
- [ ] Analysis report generated automatically

## 6. Success Criteria

- **Functionality**: All comparisons execute without errors on both implementations
- **Accuracy**: Metrics match manual calculations and are mathematically correct
- **Visualization**: Plots clearly show prediction differences in 2D space (PCA and LDA)
- **Performance**: Benchmarking captures meaningful timing differences
- **Code Quality**: All modules ≤ 200 lines, properly structured as Python package
- **Usability**: Single command runs full analysis: `uv run python -m analysis`
- **Output**: Complete report generated in `docs/analysis_report.md`

## 7. Non-Functional Requirements

- **Performance**: Analysis completes in < 10 seconds
- **Portability**: Works on Windows, macOS, Linux
- **Maintainability**: Clear separation of concerns, reusable components
- **Scalability**: Can be extended for additional classifiers/datasets

## 8. Out of Scope

- Real-time interactive dashboards
- Web UI or REST API
- Hyperparameter tuning
- Cross-validation analysis
- Other classification algorithms beyond Naive Bayes
- Custom dataset support (only Iris dataset)

## 9. Implementation Notes

### 9.1 Data Consistency
- Both implementations must use identical:
  - Random seed (42)
  - Train/test split (75/25)
  - Stratification strategy
- This ensures fair comparison

### 9.2 PCA vs LDA

**PCA (Principal Component Analysis)**:
- Unsupervised technique
- Finds directions of maximum variance
- Good for general visualization
- Preserves global structure
- Use `sklearn.decomposition.PCA`

**LDA (Linear Discriminant Analysis)**:
- Supervised technique
- Finds directions that maximize class separation
- Better for classification visualization
- Shows discriminative features
- Use `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`

### 9.3 Error Metrics

**Classification Errors**:
- Misclassification: `predicted_class ≠ true_class`
- For MSE calculation, treat class labels as numeric (0, 1, 2)
- MSE = mean((y_pred - y_true)²)
- RMSE = sqrt(MSE)
- MAE = mean(|y_pred - y_true|)

### 9.4 Runtime Measurement

Use `time.perf_counter()` for accurate timing:
```python
start = time.perf_counter()
# ... code to benchmark ...
end = time.perf_counter()
elapsed = end - start
```

### 9.5 Module Execution

The analysis package should be runnable as:
```bash
uv run python -m analysis
```

This requires `__main__.py` in the analysis package.

## 10. Report Structure Template

```markdown
# Naive Bayes Implementation Analysis Report

Generated: [timestamp]

## Executive Summary
Brief overview of findings comparing NumPy and scikit-learn implementations.

## 1. Dataset Information
- Dataset: Iris
- Total samples: 150
- Training samples: 112
- Test samples: 38
- Features: 4
- Classes: 3

## 2. Accuracy Metrics

| Metric | NumPy | scikit-learn |
|--------|-------|--------------|
| Overall Accuracy | X.XXXX | X.XXXX |
| Class 0 Precision | X.XXXX | X.XXXX |
| Class 1 Precision | X.XXXX | X.XXXX |
| Class 2 Precision | X.XXXX | X.XXXX |

## 3. Error Analysis

| Metric | NumPy | scikit-learn |
|--------|-------|--------------|
| MSE (Average Error²) | X.XXXX | X.XXXX |
| RMSE | X.XXXX | X.XXXX |
| MAE | X.XXXX | X.XXXX |
| Misclassifications | X | X |

## 4. Runtime Performance

| Phase | NumPy (ms) | scikit-learn (ms) | Speedup |
|-------|-----------|-------------------|---------|
| Training | XXX.XX ± YY.YY | XXX.XX ± YY.YY | X.XXx |
| Inference | XXX.XX ± YY.YY | XXX.XX ± YY.YY | X.XXx |
| Total | XXX.XX ± YY.YY | XXX.XX ± YY.YY | X.XXx |

## 5. Visual Analysis

### PCA Visualization
- 2D projection using first 2 principal components
- Explained variance: XX.X%
- See: `output/plots/pca_numpy.png` and `output/plots/pca_sklearn.png`

### LDA Visualization
- 2D projection using linear discriminants
- See: `output/plots/lda_numpy.png` and `output/plots/lda_sklearn.png`

## 6. Conclusions

[Key findings and recommendations]
```

## 11. Implementation Phases

**Phase 1**: Project setup
- Create `pyproject.toml` for UV
- Set up `analysis/` package structure
- Create `__init__.py` files

**Phase 2**: Core comparison (`comparison.py`)
- Refactor existing code into reusable functions
- Implement metric calculations
- Ensure data consistency

**Phase 3**: Visualization (`visualizer.py`)
- Implement PCA plotting
- Implement LDA plotting
- Create error visualization

**Phase 4**: Benchmarking (`benchmarker.py`)
- Implement timing logic
- Statistical analysis of timings

**Phase 5**: Reporting (`reporter.py`)
- Markdown report generation
- Format tables and metrics

**Phase 6**: Integration (`__main__.py`)
- Orchestrate all components
- Error handling
- Output management

**Phase 7**: Testing and documentation
- Test on clean environment
- Update `CLAUDE.md`
- Verify all deliverables

## 12. Command Reference

After implementation, the following commands should work:

```bash
# Setup environment
uv sync

# Run full analysis
uv run python -m analysis

# Or with explicit Python
uv run python -m analysis

# Run individual modules (if needed)
uv run python -m analysis.comparison
uv run python -m analysis.visualizer
```

## 13. Questions & Clarifications

- Should plots be saved as PNG, PDF, or both?
- Should the tool support CLI arguments for customization?
- Any specific styling requirements for plots?
- Should we generate an HTML report in addition to markdown?
