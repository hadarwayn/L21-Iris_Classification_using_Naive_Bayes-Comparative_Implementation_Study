# Product Requirements Document (PRD)
## Iris Flower Classification using Naive Bayes: Comparative Implementation Study

**Version:** 1.0
**Date:** December 10, 2025
**Status:** Planning Phase
**Project Type:** Machine Learning Classification & Comparative Analysis
**Estimated Development Time:** 16-20 hours

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Target Users & Applications](#target-users--applications)
4. [Functional Requirements](#functional-requirements)
5. [Technical Requirements](#technical-requirements)
6. [Success Criteria](#success-criteria)
7. [Constraints & Assumptions](#constraints--assumptions)
8. [Learning Objectives](#learning-objectives)
9. [Mathematical Foundations](#mathematical-foundations)
10. [Implementation Comparison Strategy](#implementation-comparison-strategy)
11. [Deliverables](#deliverables)
12. [Risk Management](#risk-management)

---

## 1. Executive Summary {#executive-summary}

### 1.1 Project Vision
Create a comprehensive, production-quality machine learning project that implements Gaussian Naive Bayes classification for the Iris flower dataset using **two parallel approaches**: a from-scratch NumPy implementation and scikit-learn's library implementation. The project will serve as both a functional classifier and an educational resource demonstrating the mathematical principles, implementation details, and performance characteristics of probabilistic classification.

### 1.2 Business Value
- **Educational Excellence**: Demonstrates deep understanding of Bayesian probability theory and its practical application in machine learning
- **Implementation Transparency**: Shows how mathematical formulas translate into working code through manual NumPy implementation
- **Performance Validation**: Proves custom implementation correctness by comparing against industry-standard scikit-learn
- **Professional Portfolio**: Creates a showcase project meeting enterprise-level code quality and documentation standards
- **Reproducible Research**: Establishes patterns for fair, scientific comparison between different ML implementations

### 1.3 Problem Statement
Many data science students and practitioners use machine learning libraries as "black boxes" without understanding the underlying mathematics and algorithms. This project bridges the gap between theory and practice by implementing Naive Bayes from first principles, then validating against a trusted library implementation, while maintaining professional software engineering standards.

### 1.4 Why This Matters
Understanding the internals of ML algorithms is critical for:
- **Debugging**: Knowing what can go wrong and why
- **Customization**: Modifying algorithms for specific use cases
- **Performance**: Understanding computational complexity and optimization opportunities
- **Trust**: Validating that library implementations work as documented
- **Career Development**: Demonstrating deep technical knowledge to employers

---

## 2. Project Overview {#project-overview}

### 2.1 Project Identity
- **Project Name:** L21 - Iris Classification using Naive Bayes: Comparative Implementation Study
- **Repository Name:** `L21-Iris-Naive-Bayes-Comparison`
- **One-Line Description:** Dual implementation Naive Bayes classifier for Iris flowers, comparing manual NumPy implementation against scikit-learn with comprehensive performance metrics and educational visualizations

### 2.2 Core Objectives
1. **Implement Gaussian Naive Bayes from scratch** using only NumPy, following Bayes' theorem and Gaussian probability distributions
2. **Implement scikit-learn wrapper** with identical data splits and evaluation methodology
3. **Perform rigorous comparison** measuring accuracy, precision, recall, F1-score, confusion matrices, and runtime performance
4. **Generate educational visualizations** showing feature distributions, decision boundaries, and classification results
5. **Produce comprehensive documentation** explaining both the mathematics and the implementation at a level accessible to 15-year-old students

### 2.3 Dataset Overview
- **Source:** Iris flower dataset (Iris.csv from Kaggle)
- **Samples:** 150 instances
- **Features:** 4 continuous measurements
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes:** 3 species
  - Iris-setosa (Class 0)
  - Iris-versicolor (Class 1)
  - Iris-virginica (Class 2)
- **Class Distribution:** Balanced (50 samples per class)
- **Data Quality:** No missing values, well-separated classes

### 2.4 Key Success Metrics (Preview)
- Both implementations achieve **≥90% accuracy** on test set
- Prediction agreement between implementations **≥95%**
- All Python files **≤150 lines** (strict modular design)
- **Minimum 3 different visualizations** (feature distributions, confusion matrices, comparison charts)
- **Minimum 3 experiment runs** with different configurations documented
- **100% reproducibility** with fixed random seeds and version-locked dependencies

---

## 3. Target Users & Applications {#target-users--applications}

### 3.1 Primary Users

#### User Persona 1: AI Development Course Student
- **Profile:** 15-18 years old, learning machine learning fundamentals
- **Goals:** Understand Naive Bayes algorithm, pass course assignment, build portfolio
- **Needs:** Clear documentation, step-by-step explanations, visual learning aids
- **Technical Level:** Beginner to intermediate Python, first exposure to ML algorithms

#### User Persona 2: Course Instructor/Reviewer
- **Profile:** Experienced AI educator or professional developer
- **Goals:** Assess student understanding, verify code quality, evaluate documentation
- **Needs:** Professional code structure, proper logging, comprehensive testing, clear metrics
- **Technical Level:** Expert in ML and software engineering best practices

#### User Persona 3: Future Students & Self-Learners
- **Profile:** Anyone learning machine learning from scratch
- **Goals:** Reference implementation, understand algorithm internals, learn from working code
- **Needs:** Well-commented code, mathematical explanations, visualizations, troubleshooting guides
- **Technical Level:** Variable (documentation must serve all levels)

### 3.2 Use Cases

#### Use Case 1: Learning Naive Bayes Algorithm
**Scenario:** A student needs to understand how Naive Bayes classification works at the mathematical level, not just the API level.

**User Journey:**
1. Read the README's mathematical foundations section (Bayes' theorem explained with examples)
2. Review the manual NumPy implementation in `src/naive_bayes_manual.py`
3. Follow inline comments explaining each calculation (priors, likelihoods, posteriors)
4. Run the code and see intermediate outputs via logging
5. Compare with scikit-learn implementation to validate understanding
6. Experiment with different parameters (Laplace smoothing, different splits)

**Success Criteria:**
- Student can explain Bayes' theorem in their own words
- Student understands conditional probability and independence assumption
- Student can identify where Gaussian distributions are used
- Student can modify the code for a different dataset

#### Use Case 2: Comparative Algorithm Analysis
**Scenario:** A researcher or advanced student wants to compare custom implementation performance against standard libraries.

**User Journey:**
1. Run both implementations on identical train/test splits
2. Compare accuracy, precision, recall, F1-score metrics side-by-side
3. Analyze confusion matrices to understand error patterns
4. Review runtime performance benchmarks (training time, prediction time)
5. Examine parameter estimates (means, variances, priors) from both implementations
6. Identify any discrepancies and understand their sources (numerical precision, regularization, etc.)

**Success Criteria:**
- Clear side-by-side metric comparison tables in results
- Quantified performance differences (speedup factors, accuracy deltas)
- Visual comparison graphs showing runtime and accuracy
- Analysis section explaining any differences found

#### Use Case 3: Portfolio Project for Job Applications
**Scenario:** A student wants to showcase ML understanding and software engineering skills to potential employers.

**User Journey:**
1. Clone repository and set up environment following README instructions
2. Run complete pipeline and generate all results
3. Review professional project structure, documentation, and code quality
4. Understand design decisions (modularity, logging, testing, visualization)
5. Present project in interview, walking through implementation details
6. Demonstrate ability to explain both theory and practice

**Success Criteria:**
- Repository demonstrates professional software engineering (structure, docs, tests)
- Code quality meets enterprise standards (type hints, docstrings, modularity)
- Clear evidence of understanding ML fundamentals (not just library API usage)
- Reproducible results with comprehensive documentation
- Visual results impressive for non-technical stakeholders

#### Use Case 4: Educational Reference for Teaching
**Scenario:** A teacher uses this project as a teaching tool in a machine learning course.

**User Journey:**
1. Walk students through the mathematical foundations in README
2. Show manual implementation as live coding example
3. Run visualizations to explain feature distributions and decision boundaries
4. Use comparison results to teach validation methodology
5. Assign students to extend the project (add new features, try different datasets)
6. Grade based on established quality criteria

**Success Criteria:**
- Documentation clear enough for self-study
- Code simple enough to understand in one reading session
- Visualizations suitable for classroom projection
- Extensible architecture for student modifications
- Troubleshooting guide for common student errors

### 3.3 Example Applications

#### Application 1: Medical Diagnosis
**Domain:** Healthcare - Disease Classification
**Scenario:** Given patient symptoms (continuous measurements like temperature, blood pressure), classify disease type
**Connection to Iris:** Similar multi-class classification problem with continuous features
**Learning Value:** Understand how probabilistic classifiers handle uncertainty in medical decision-making

#### Application 2: Quality Control in Manufacturing
**Domain:** Industrial - Product Defect Classification
**Scenario:** Given product measurements (dimensions, weights, material properties), classify as defect type or acceptable
**Connection to Iris:** Multi-class classification of physical measurements
**Learning Value:** Understand how Naive Bayes handles measurement variability

#### Application 3: Document Classification
**Domain:** Natural Language Processing - Email Spam Detection
**Scenario:** Given email features (word frequencies, header info), classify as spam, ham, or promotional
**Connection to Iris:** Multi-class classification with independent features
**Learning Value:** See how Naive Bayes' independence assumption applies to text data

#### Application 4: Species Identification
**Domain:** Biology - Wildlife Classification
**Scenario:** Given animal measurements (size, weight, observed characteristics), identify species
**Connection to Iris:** Direct application - classifying biological specimens by measurements
**Learning Value:** Real-world botanical/zoological classification task

---

## 4. Functional Requirements {#functional-requirements}

### 4.1 Data Management (Priority: P0 - Critical)

#### FR-1.1: Dataset Loading and Validation
**Description:** Load the Iris dataset from CSV file and validate data integrity.

**Requirements:**
- Load `Iris.csv` from `data/` directory using pandas
- Validate schema: 4 numeric features + 1 categorical target
- Verify 150 samples with no missing values
- Check class balance (50 samples per class)
- Convert species names to numeric labels (0, 1, 2)
- Log dataset statistics (shape, dtypes, class distribution)

**Input:**
- File path: `data/Iris.csv`
- CSV format: `Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species`

**Output:**
- NumPy array X: shape (150, 4), dtype float64
- NumPy array y: shape (150,), dtype int64
- Log entries confirming successful load

**Error Handling:**
- File not found: Clear error message with expected path
- Schema mismatch: List expected vs. actual columns
- Missing values: Report count and locations
- Invalid values: Report rows with non-numeric or invalid data

**Acceptance Criteria:**
- ✅ Loads exactly 150 samples
- ✅ Extracts exactly 4 features
- ✅ Maps 3 species to integers 0, 1, 2
- ✅ Validates no NaN or infinite values
- ✅ Logs dataset statistics

#### FR-1.2: Train-Test Split with Stratification
**Description:** Split data into training and testing sets while maintaining class distribution.

**Requirements:**
- Use 80/20 split (120 training, 30 testing samples)
- Apply stratified split to maintain class balance in both sets
- Use fixed random seed (42) for reproducibility
- Validate split maintains class proportions (±5% tolerance)
- Log split statistics (training size, test size, class distribution in each)

**Input:**
- X: feature array (150, 4)
- y: label array (150,)
- test_size: 0.20
- random_state: 42

**Output:**
- X_train: (120, 4) training features
- X_test: (30, 4) test features
- y_train: (120,) training labels
- y_test: (30,) test labels

**Implementation:**
- Use `sklearn.model_selection.train_test_split` with `stratify=y`
- Verify class distribution in training set ≈ 40:40:40
- Verify class distribution in test set ≈ 10:10:10

**Acceptance Criteria:**
- ✅ Training set: 120 samples (80%)
- ✅ Test set: 30 samples (20%)
- ✅ Each class represented proportionally in both sets
- ✅ Same split achieved with random_state=42 across runs
- ✅ Split statistics logged

#### FR-1.3: Data Preprocessing and Normalization
**Description:** Optional preprocessing steps for improved numerical stability.

**Requirements:**
- Implement optional feature scaling (standardization: mean=0, std=1)
- Calculate scaling parameters from training set only (prevent data leakage)
- Apply same transformation to test set
- Log scaling parameters (means, standard deviations)
- Provide flag to enable/disable scaling for experiments

**Input:**
- X_train: (120, 4)
- X_test: (30, 4)
- normalize: boolean flag

**Output:**
- X_train_scaled: (120, 4) if normalize=True
- X_test_scaled: (30, 4) if normalize=True
- scaling_params: dict with 'mean' and 'std' arrays

**Note:** For Naive Bayes, scaling may not improve performance due to per-class mean/variance estimation, but implementation is included for completeness.

**Acceptance Criteria:**
- ✅ If normalize=True, training data has mean ≈ 0, std ≈ 1 per feature
- ✅ Test data transformed using training set parameters
- ✅ Original data preserved (no in-place modification)
- ✅ Scaling parameters logged

### 4.2 NumPy Implementation (Priority: P0 - Critical)

#### FR-2.1: Gaussian Naive Bayes Manual Implementation
**Description:** Implement Gaussian Naive Bayes classifier from mathematical first principles using only NumPy.

**Mathematical Basis:**
```
Training:
  For each class c:
    Prior: P(c) = count(c) / total_samples
    For each feature f:
      Mean: μ[c][f] = mean(X[y==c][:, f])
      Variance: σ²[c][f] = var(X[y==c][:, f]) + epsilon

Prediction:
  For each test sample x:
    For each class c:
      log_posterior[c] = log(P(c))
      For each feature i:
        log_likelihood = log(gaussian_pdf(x[i] | μ[c][i], σ²[c][i]))
        log_posterior[c] += log_likelihood
    prediction = argmax(log_posterior)
```

**Requirements:**
- Create `GaussianNaiveBayesNumPy` class with `fit()` and `predict()` methods
- Calculate class priors: P(y = c)
- Calculate per-class feature means: μ[c][f]
- Calculate per-class feature variances: σ²[c][f]
- Add numerical stability epsilon (1e-9) to variances
- Use log probabilities throughout to prevent numerical underflow
- Implement Gaussian PDF: `(1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))`
- Support multi-class classification (3 classes)
- Store learned parameters for inspection

**Class Interface:**
```python
class GaussianNaiveBayesNumPy:
    def __init__(self, epsilon: float = 1e-9):
        """Initialize with numerical stability parameter."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model on X, y."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for X."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
```

**Acceptance Criteria:**
- ✅ Achieves ≥90% accuracy on Iris test set
- ✅ Uses only NumPy (no scikit-learn in core algorithm)
- ✅ Implements Gaussian PDF correctly
- ✅ Uses log probabilities (no numerical underflow)
- ✅ Handles all 3 classes
- ✅ Logs learned parameters (priors, means, variances)
- ✅ Module ≤150 lines

#### FR-2.2: Training Process with Parameter Logging
**Description:** Train the manual Naive Bayes model and log all learned parameters.

**Requirements:**
- Call `fit(X_train, y_train)` to train model
- Calculate and store:
  - Class priors: (3,) array
  - Feature means: (3, 4) array
  - Feature variances: (3, 4) array
  - Class labels: (3,) array
- Log learned parameters in human-readable format:
  ```
  Class 0 (Iris-setosa):
    Prior: 0.3333
    Feature 0 (Sepal Length): μ=5.01, σ²=0.12
    Feature 1 (Sepal Width): μ=3.42, σ²=0.14
    ...
  ```
- Verify parameters make intuitive sense (positive variances, priors sum to 1)

**Acceptance Criteria:**
- ✅ All parameters calculated correctly
- ✅ Priors sum to 1.0 (within floating point tolerance)
- ✅ All variances > 0
- ✅ Parameters logged with class and feature names
- ✅ Training completes in < 1 second

#### FR-2.3: Prediction with Detailed Logging
**Description:** Make predictions on test set and log prediction process for educational purposes.

**Requirements:**
- Call `predict(X_test)` to get predictions
- For first 3 test samples, log detailed calculation:
  - Input features
  - Log probabilities for each class
  - Predicted class and confidence
- Generate full predictions for all test samples
- Calculate and log test accuracy
- Return predictions as NumPy array

**Output Example:**
```
Sample 1: [5.9, 3.0, 5.1, 1.8]
  Class 0: log_prob = -12.34
  Class 1: log_prob = -8.56
  Class 2: log_prob = -3.21 ← PREDICTED (Confidence: 0.89)
Test Accuracy: 93.33% (28/30 correct)
```

**Acceptance Criteria:**
- ✅ Predictions shape: (30,)
- ✅ All predictions in range [0, 1, 2]
- ✅ Accuracy ≥ 90%
- ✅ Detailed logging for educational clarity
- ✅ Prediction completes in < 0.5 seconds

#### FR-2.4: Feature Distribution Visualization
**Description:** Generate visualizations showing how features are distributed across classes.

**Requirements:**
- Create 2x2 subplot grid (one subplot per feature)
- For each feature:
  - Plot histogram for each class (3 overlaid histograms)
  - Use distinct colors: Class 0 (red), Class 1 (green), Class 2 (blue)
  - Add semi-transparency (alpha=0.5) for overlap visibility
  - Mark class means with vertical dashed lines
  - Label axes: feature name and count
- Add overall title: "Feature Distributions by Class"
- Add legend identifying classes
- Save to `results/graphs/feature_distributions_numpy.png` at 300 DPI

**Purpose:** Show how Naive Bayes leverages different feature distributions for classification.

**Acceptance Criteria:**
- ✅ All 4 features visualized
- ✅ Clear visual separation between classes
- ✅ Means marked for reference
- ✅ High-resolution PNG saved
- ✅ Educational value: students can see why algorithm works

### 4.3 Scikit-learn Implementation (Priority: P0 - Critical)

#### FR-3.1: GaussianNB Wrapper Implementation
**Description:** Create a wrapper around scikit-learn's GaussianNB for fair comparison.

**Requirements:**
- Create `GaussianNaiveBayesSKlearn` class wrapping `sklearn.naive_bayes.GaussianNB`
- Use identical train/test split as NumPy implementation
- Set any applicable hyperparameters to match NumPy implementation (e.g., var_smoothing for epsilon)
- Provide same interface as NumPy class (`fit`, `predict`, `score`)
- Extract and log learned parameters from sklearn model

**Class Interface:**
```python
class GaussianNaiveBayesSKlearn:
    def __init__(self, var_smoothing: float = 1e-9):
        """Initialize sklearn GaussianNB with parameters."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train sklearn model."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using sklearn."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""

    def get_params(self) -> dict:
        """Extract learned parameters for comparison."""
```

**Acceptance Criteria:**
- ✅ Uses `sklearn.naive_bayes.GaussianNB`
- ✅ Same interface as NumPy implementation
- ✅ Achieves ≥90% accuracy
- ✅ Can extract learned parameters (theta, sigma, class_prior)
- ✅ Module ≤150 lines

#### FR-3.2: Parameter Extraction and Comparison
**Description:** Extract learned parameters from sklearn model for comparison with NumPy implementation.

**Requirements:**
- After training, extract:
  - `model.class_prior_`: prior probabilities
  - `model.theta_`: feature means per class
  - `model.var_`: feature variances per class
- Log parameters in same format as NumPy implementation
- Calculate parameter differences (NumPy - sklearn):
  - Mean absolute difference in priors
  - Mean absolute difference in means
  - Mean absolute difference in variances
- Log comparison results

**Expected Outcome:** Parameters should match closely (differences < 0.01) when using same epsilon/var_smoothing.

**Acceptance Criteria:**
- ✅ All sklearn parameters extracted
- ✅ Parameters logged in comparable format
- ✅ Differences calculated and logged
- ✅ Small differences indicate correct NumPy implementation

#### FR-3.3: Comprehensive Evaluation Metrics
**Description:** Generate detailed evaluation metrics using sklearn's tools.

**Requirements:**
- Generate classification report (precision, recall, F1-score per class)
- Compute confusion matrix
- Calculate overall accuracy
- Calculate macro-averaged and weighted-averaged metrics
- Log all metrics in structured format
- Compare metrics with NumPy implementation

**Metrics to Calculate:**
- Accuracy: (TP + TN) / Total
- Precision per class: TP / (TP + FP)
- Recall per class: TP / (TP + FN)
- F1-score per class: 2 × (Precision × Recall) / (Precision + Recall)
- Support per class: actual number of samples
- Macro average: unweighted mean of per-class metrics
- Weighted average: class-size-weighted mean of per-class metrics

**Acceptance Criteria:**
- ✅ Classification report generated
- ✅ Confusion matrix (3x3) computed
- ✅ All metrics calculated and logged
- ✅ Metrics compared with NumPy implementation
- ✅ Results formatted for inclusion in README

### 4.4 Comparison and Analysis (Priority: P0 - Critical)

#### FR-4.1: Prediction-Level Comparison
**Description:** Compare predictions element-by-element between NumPy and sklearn implementations.

**Requirements:**
- Compare predictions on test set: `y_pred_numpy` vs `y_pred_sklearn`
- Calculate agreement rate: percentage of identical predictions
- Identify disagreement indices: samples where predictions differ
- For disagreements, log:
  - Sample features
  - True label
  - NumPy prediction and confidence
  - Sklearn prediction and confidence
  - Both models' log probabilities for all classes
- Analyze why disagreements occur (numerical precision, regularization differences, etc.)

**Output:**
```
Prediction Agreement: 96.67% (29/30 identical)
Disagreements:
  Sample 15: True=1, NumPy→1 (0.89), Sklearn→2 (0.78)
    Features: [6.1, 2.8, 4.7, 1.2]
    Analysis: Close decision boundary, small variance difference
```

**Acceptance Criteria:**
- ✅ Agreement rate ≥95%
- ✅ All disagreements documented
- ✅ Analysis provided for each disagreement
- ✅ Results logged and saved to file

#### FR-4.2: Metric-Level Comparison
**Description:** Compare accuracy, precision, recall, and F1-score between implementations.

**Requirements:**
- Generate side-by-side comparison table:
  ```
  Metric              | NumPy    | Sklearn  | Difference
  --------------------|----------|----------|------------
  Accuracy            | 0.9333   | 0.9667   | -0.0334
  Macro Avg Precision | 0.9400   | 0.9700   | -0.0300
  Macro Avg Recall    | 0.9333   | 0.9667   | -0.0334
  Macro Avg F1        | 0.9350   | 0.9680   | -0.0330
  ```
- Calculate differences for all metrics
- Visualize metric comparison as bar chart
- Analyze whether differences are statistically significant
- Provide interpretation of differences

**Acceptance Criteria:**
- ✅ All metrics compared side-by-side
- ✅ Differences calculated and displayed
- ✅ Comparison table saved as CSV
- ✅ Bar chart saved as PNG
- ✅ Analysis provided

#### FR-4.3: Confusion Matrix Comparison
**Description:** Generate and compare confusion matrices visually.

**Requirements:**
- Create 1x2 subplot showing confusion matrices side-by-side
- Left subplot: NumPy confusion matrix
- Right subplot: Sklearn confusion matrix
- Use heatmap visualization with annotations
- Color scale: lighter for lower values, darker for higher
- Label axes: Predicted vs. True classes
- Add overall title: "Confusion Matrix Comparison"
- Save to `results/graphs/confusion_matrix_comparison.png`

**Purpose:** Visually compare error patterns between implementations.

**Acceptance Criteria:**
- ✅ Both confusion matrices displayed
- ✅ Cell values annotated
- ✅ Clear color scale
- ✅ High-resolution PNG saved
- ✅ Easy to identify differences visually

#### FR-4.4: Runtime Performance Comparison
**Description:** Benchmark training and prediction time for both implementations.

**Requirements:**
- Measure training time for each implementation (multiple runs)
- Measure prediction time for each implementation (multiple runs)
- Calculate statistics: mean, std, min, max
- Run minimum 10 iterations for statistical significance
- Control for environmental factors (same machine, same data)
- Calculate speedup factor: `time_sklearn / time_numpy` or vice versa
- Generate runtime comparison bar chart
- Log detailed timing results

**Expected Results:**
- Sklearn likely faster (optimized C code)
- NumPy implementation should be reasonable (< 5x slower)

**Output:**
```
Runtime Comparison (mean ± std, n=10):
  Training Time:
    NumPy:   2.34 ± 0.12 ms
    Sklearn: 1.45 ± 0.08 ms
    Speedup: 1.61x (sklearn faster)

  Prediction Time:
    NumPy:   0.89 ± 0.05 ms
    Sklearn: 0.52 ± 0.03 ms
    Speedup: 1.71x (sklearn faster)
```

**Acceptance Criteria:**
- ✅ Both training and prediction times measured
- ✅ Multiple runs for statistical validity
- ✅ Mean, std, min, max calculated
- ✅ Speedup factors computed
- ✅ Bar chart visualization saved
- ✅ Results logged and saved to CSV

#### FR-4.5: Parameter-Level Comparison
**Description:** Compare learned parameters (priors, means, variances) between implementations.

**Requirements:**
- Extract parameters from both models
- Calculate element-wise differences
- Compute summary statistics of differences:
  - Mean absolute error (MAE)
  - Maximum absolute error
  - Root mean squared error (RMSE)
- Visualize parameter comparisons as scatter plots
  - X-axis: NumPy parameter values
  - Y-axis: Sklearn parameter values
  - Ideal: all points on y=x line
- Analyze and explain any systematic differences

**Purpose:** Verify that NumPy implementation correctly learns the same parameters as sklearn.

**Acceptance Criteria:**
- ✅ All parameters compared
- ✅ Differences quantified with MAE, RMSE
- ✅ Scatter plots generated
- ✅ Small differences (< 0.01 for means/vars, < 0.001 for priors)
- ✅ Analysis provided for any large differences

### 4.5 Visualization and Reporting (Priority: P1 - High)

#### FR-5.1: Results Visualization Suite
**Description:** Generate comprehensive visualizations for README and analysis.

**Requirements:**
- **Graph 1:** Feature distributions by class (2x2 grid, histograms)
- **Graph 2:** Confusion matrix comparison (1x2 side-by-side heatmaps)
- **Graph 3:** Metric comparison bar chart (accuracy, precision, recall, F1)
- **Graph 4:** Runtime comparison bar chart (training time, prediction time)
- **Graph 5:** Parameter scatter plot (NumPy vs sklearn learned parameters)
- All graphs saved to `results/graphs/` directory
- All graphs saved at 300 DPI PNG format
- All graphs have clear titles, axis labels, and legends
- Use colorblind-friendly color palette

**Acceptance Criteria:**
- ✅ Minimum 5 different visualizations generated
- ✅ All graphs high-resolution (300 DPI)
- ✅ Clear, professional styling
- ✅ Legends and labels present
- ✅ Saved to results/graphs/

#### FR-5.2: Experiment Results Documentation
**Description:** Run and document minimum 3 experiments with different configurations.

**Requirements:**
- **Experiment 1:** Baseline (80/20 split, seed=42, no scaling)
- **Experiment 2:** Different split (70/30 split, seed=42, no scaling)
- **Experiment 3:** With feature scaling (80/20 split, seed=42, standardized features)
- For each experiment:
  - Save metrics to `results/examples/runX/metrics.json`
  - Save predictions to `results/examples/runX/predictions.csv`
  - Save visualizations to `results/examples/runX/`
  - Document configuration in `results/examples/runX/config.yaml`
- Generate summary table comparing all experiments

**Purpose:** Demonstrate how different configurations affect results.

**Acceptance Criteria:**
- ✅ 3 experiments completed and documented
- ✅ Each experiment has full results saved
- ✅ Summary comparison table generated
- ✅ Analysis of configuration effects written

#### FR-5.3: Performance Metrics Summary
**Description:** Create comprehensive summary of all performance metrics.

**Requirements:**
- Generate `results/tables/metrics_summary.csv` with all runs
- Include columns: Run, Implementation, Accuracy, Precision, Recall, F1, Training_Time, Prediction_Time
- Calculate aggregate statistics (mean, std across runs)
- Generate markdown table for README inclusion
- Create metric trends visualization (if multiple runs)

**Acceptance Criteria:**
- ✅ CSV file with all metrics
- ✅ Markdown table generated
- ✅ Statistics calculated
- ✅ Ready for README inclusion

### 4.6 Orchestration and Execution (Priority: P0 - Critical)

#### FR-6.1: Main Pipeline Implementation
**Description:** Implement main.py as single entry point orchestrating full pipeline.

**Requirements:**
- Entry point: `python main.py` (no arguments required for default run)
- Optional arguments:
  - `--config`: path to configuration file (default: `config/settings.yaml`)
  - `--run-name`: name for this experiment run (default: auto-generated timestamp)
  - `--verbose`: enable detailed logging
- Pipeline steps:
  1. Load configuration
  2. Setup logging (ring buffer)
  3. Load and split data
  4. Train NumPy model
  5. Train sklearn model
  6. Generate predictions (both models)
  7. Calculate all metrics
  8. Run performance benchmarks
  9. Generate visualizations
  10. Save all results
  11. Print summary report
- Error handling at each step with clear messages
- Progress indicators (Step 1/10: Loading data...)

**Acceptance Criteria:**
- ✅ Single command execution
- ✅ Clear progress output
- ✅ Error handling at all steps
- ✅ All results saved to results/ directory
- ✅ Summary printed at end

#### FR-6.2: Configuration Management
**Description:** Implement flexible configuration system using YAML.

**Requirements:**
- Create `config/settings.yaml` with all configurable parameters:
  ```yaml
  data:
    path: "data/Iris.csv"
    test_size: 0.20
    random_seed: 42
    normalize: false

  models:
    numpy:
      epsilon: 1.0e-9
    sklearn:
      var_smoothing: 1.0e-9

  evaluation:
    benchmark_runs: 10
    metrics: ["accuracy", "precision", "recall", "f1"]

  visualization:
    dpi: 300
    style: "seaborn-v0_8"

  output:
    results_dir: "results"
    save_predictions: true
    save_models: false
  ```
- Load configuration in main.py
- Validate configuration values
- Log loaded configuration
- Allow command-line override of key parameters

**Acceptance Criteria:**
- ✅ YAML configuration file
- ✅ All parameters configurable
- ✅ Configuration validated
- ✅ Sensible defaults provided
- ✅ Easy to modify for experiments

---

## 5. Technical Requirements {#technical-requirements}

### 5.1 Environment and Tools

#### 5.1.1 Python Version
- **Required:** Python 3.10 or higher
- **Rationale:** Type hints, match statements, improved error messages
- **Verification:** `python --version` must show ≥3.10.0

#### 5.1.2 Virtual Environment Management
- **Tool:** UV (ultra-fast Python package manager)
- **Installation (WSL/Linux/Mac):**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Installation (Windows PowerShell):**
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Environment Setup:**
  ```bash
  uv venv
  source .venv/bin/activate  # WSL/Linux/Mac
  .venv\Scripts\activate     # Windows
  uv pip install -r requirements.txt
  ```
- **Requirement:** Must include `venv/.gitkeep` indicator folder in repository

#### 5.1.3 Operating System Support
- **Primary:** Windows 11 with WSL (Windows Subsystem for Linux)
- **Secondary:** Linux (Ubuntu 20.04+), macOS (10.15+)
- **Documentation:** Provide instructions for both WSL and Windows PowerShell

### 5.2 Core Technologies

#### 5.2.1 Mandatory Libraries
- **NumPy (≥1.24.0):**
  - Array operations and vectorized computation
  - Random number generation with fixed seeds
  - Statistical functions (mean, var, sum)
  - Linear algebra operations
  - **Critical:** No basic Python loops in core algorithms

- **Pandas (≥2.0.0):**
  - CSV data loading
  - DataFrame operations for data inspection
  - Data validation and preprocessing

- **Scikit-learn (≥1.3.0):**
  - GaussianNB implementation for comparison
  - Train-test split utilities
  - Evaluation metrics (accuracy_score, classification_report, confusion_matrix)
  - Optional: StandardScaler for feature normalization

- **Matplotlib (≥3.7.0):**
  - Visualization generation
  - Histograms, heatmaps, bar charts, scatter plots
  - High-resolution PNG export

- **PyYAML (≥6.0):**
  - Configuration file parsing
  - Parameter management

- **python-dotenv (≥1.0.0):**
  - Environment variable management
  - Secret management (.env file)

#### 5.2.2 Development Tools
- **Black (≥23.0.0):** Code formatting
- **Pytest (≥7.4.0):** Unit testing framework (optional)
- **MyPy (≥1.5.0):** Static type checking (optional)

#### 5.2.3 Version Pinning
- Use exact versions in `requirements.txt`
- Format: `package==X.Y.Z`
- Generate via: `uv pip freeze > requirements.txt`
- Example:
  ```
  numpy==1.24.3
  pandas==2.0.2
  scikit-learn==1.3.0
  matplotlib==3.7.1
  pyyaml==6.0
  python-dotenv==1.0.0
  ```

### 5.3 Code Quality Standards

#### 5.3.1 File Length Limit
- **STRICT RULE:** Maximum 150 lines per Python file
- **Enforcement:** Automatic check in pre-commit
- **Rationale:** Forces modularity, single responsibility, readability
- **Solution:** Split large files into focused modules

#### 5.3.2 Type Hints
- **Requirement:** All function signatures must have type hints
- **Coverage:** Parameters, return values, class attributes
- **Example:**
  ```python
  def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
      """Calculate classification accuracy."""
      return np.mean(y_true == y_pred)
  ```

#### 5.3.3 Docstrings
- **Requirement:** All modules, classes, and functions must have docstrings
- **Format:** Google style docstrings
- **Components:**
  - One-line summary
  - Detailed description (if needed)
  - Args section with types and descriptions
  - Returns section with type and description
  - Raises section (if applicable)
  - Example section (if helpful)
  - Note section (if needed)

- **Example:**
  ```python
  def gaussian_pdf(x: float, mean: float, var: float) -> float:
      """
      Calculate Gaussian probability density function value.

      The Gaussian (normal) distribution is defined as:
      P(x | μ, σ²) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

      Args:
          x: Input value to evaluate
          mean: Mean (μ) of the Gaussian distribution
          var: Variance (σ²) of the Gaussian distribution

      Returns:
          Probability density value at x (always positive)

      Example:
          >>> gaussian_pdf(0, 0, 1)
          0.3989422804014327  # 1/√(2π)

      Note:
          Small epsilon is added to variance to prevent division by zero.
      """
      return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))
  ```

#### 5.3.4 Code Comments
- **Target Audience:** 15-year-old with basic Python knowledge
- **Focus:** Explain WHY, not WHAT
- **Style:** Complete sentences, proper grammar
- **Example:**
  ```python
  # We use log probabilities to prevent numerical underflow.
  # Multiplying many small probabilities (e.g., 0.001 × 0.001 × 0.001)
  # can result in values too small for float64 to represent.
  # Log probabilities turn multiplication into addition: log(a×b) = log(a) + log(b)
  log_posterior = np.log(prior) + np.sum(log_likelihoods)
  ```

#### 5.3.5 Naming Conventions
- **Variables:** snake_case (e.g., `class_priors`, `feature_means`)
- **Functions:** snake_case (e.g., `calculate_accuracy`, `plot_confusion_matrix`)
- **Classes:** PascalCase (e.g., `GaussianNaiveBayesNumPy`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `RANDOM_SEED`, `NUM_CLASSES`)
- **Private:** Leading underscore (e.g., `_gaussian_pdf`, `_log_likelihood`)
- **Descriptive:** Avoid abbreviations, use full words (e.g., `precision` not `prec`)

#### 5.3.6 Error Handling
- **Requirement:** Validate all inputs, handle expected errors gracefully
- **Approach:**
  - Validate function inputs at start (type, shape, range)
  - Use informative error messages
  - Catch specific exceptions
  - Log errors before raising
- **Example:**
  ```python
  def fit(self, X: np.ndarray, y: np.ndarray) -> None:
      """Train Naive Bayes model."""
      # Validate inputs
      if not isinstance(X, np.ndarray):
          raise TypeError(f"X must be np.ndarray, got {type(X)}")
      if not isinstance(y, np.ndarray):
          raise TypeError(f"y must be np.ndarray, got {type(y)}")
      if X.shape[0] != y.shape[0]:
          raise ValueError(f"X and y must have same length: {X.shape[0]} != {y.shape[0]}")
      if X.ndim != 2:
          raise ValueError(f"X must be 2D array, got shape {X.shape}")
      if np.any(np.isnan(X)) or np.any(np.isnan(y)):
          raise ValueError("X and y cannot contain NaN values")

      # Training logic...
  ```

### 5.4 Directory Structure

#### 5.4.1 Standard Layout (MANDATORY)
```
L21-Iris-Naive-Bayes-Comparison/
├── README.md                          # Comprehensive documentation
├── main.py                            # Single entry point
├── requirements.txt                   # Exact dependency versions
├── .gitignore                         # Exclude secrets, cache, venv
├── .env.example                       # Template for environment variables
│
├── venv/                              # Virtual environment indicator
│   └── .gitkeep                       # Instructions for venv setup
│
├── src/                               # ALL source code
│   ├── __init__.py                    # Package marker (required)
│   ├── naive_bayes_manual.py          # NumPy implementation (≤150 lines)
│   ├── naive_bayes_sklearn.py         # Sklearn wrapper (≤150 lines)
│   ├── data_loader.py                 # Data loading/splitting (≤150 lines)
│   ├── evaluator.py                   # Metrics calculation (≤150 lines)
│   ├── visualizer.py                  # Visualization generation (≤150 lines)
│   └── utils/                         # Helper functions
│       ├── __init__.py                # Package marker (required)
│       ├── logger.py                  # Ring buffer logging (≤150 lines)
│       ├── validators.py              # Input validation (≤150 lines)
│       ├── helpers.py                 # Utility functions (≤150 lines)
│       └── paths.py                   # Path management (≤150 lines)
│
├── docs/                              # Documentation
│   ├── PRD.md                         # This document
│   ├── tasks.json                     # Task breakdown
│   └── API.md                         # API documentation (optional)
│
├── data/                              # Dataset storage
│   ├── Iris.csv                       # Iris dataset
│   └── README.md                      # Data source information
│
├── results/                           # ALL output files
│   ├── examples/                      # Multiple experiment runs
│   │   ├── run_1/                     # Experiment 1 results
│   │   │   ├── config.yaml            # Configuration used
│   │   │   ├── metrics.json           # All metrics
│   │   │   ├── predictions.csv        # Model predictions
│   │   │   └── visualizations/        # Run-specific graphs
│   │   ├── run_2/                     # Experiment 2 results
│   │   └── run_3/                     # Experiment 3 results
│   ├── graphs/                        # Shared visualizations
│   │   ├── feature_distributions.png
│   │   ├── confusion_matrix_comparison.png
│   │   ├── metric_comparison.png
│   │   ├── runtime_comparison.png
│   │   └── parameter_scatter.png
│   └── tables/                        # Result tables
│       ├── metrics_summary.csv
│       ├── runtime_summary.csv
│       └── parameter_comparison.csv
│
├── logs/                              # Logging output (ring buffer)
│   ├── config/                        # Log configuration
│   │   └── log_config.json            # Ring buffer settings
│   └── .gitkeep                       # Keep folder in git
│
├── config/                            # Configuration files
│   ├── settings.yaml                  # Main configuration
│   └── experiments/                   # Experiment configurations
│       ├── baseline.yaml
│       ├── different_split.yaml
│       └── with_scaling.yaml
│
└── tests/                             # Unit tests (optional)
    ├── __init__.py
    ├── test_naive_bayes_manual.py
    ├── test_naive_bayes_sklearn.py
    └── test_data_loader.py
```

#### 5.4.2 Package Structure Rules
- **Every directory containing Python code MUST have `__init__.py`**
- All imports must use relative paths (pathlib.Path)
- No hardcoded absolute paths anywhere in code
- Use utility function `get_project_root()` for path resolution

### 5.5 Performance Requirements

#### 5.5.1 Execution Time
- **Data loading:** < 1 second
- **Training (NumPy):** < 2 seconds
- **Training (Sklearn):** < 1 second
- **Prediction (NumPy):** < 0.5 seconds
- **Prediction (Sklearn):** < 0.3 seconds
- **Full pipeline:** < 15 seconds (including visualizations)
- **Benchmark runs (10 iterations):** < 30 seconds

#### 5.5.2 Memory Usage
- **Peak memory:** < 500 MB
- **Rationale:** Iris dataset is small (150 samples × 4 features)
- **Monitoring:** Log memory usage at key points
- **No memory leaks:** Verify with multiple runs

#### 5.5.3 Model Performance
- **NumPy Accuracy:** ≥90% on test set
- **Sklearn Accuracy:** ≥90% on test set
- **Agreement Rate:** ≥95% between implementations
- **Rationale:** Iris is a well-separated dataset, high accuracy is expected

### 5.6 Security Requirements

#### 5.6.1 Secret Management
- **No hardcoded secrets** in code (API keys, passwords, etc.)
- Use `.env` file for environment variables
- Include `.env.example` with template (safe values)
- `.env` must be in `.gitignore`

#### 5.6.2 Input Validation
- Validate all user inputs (file paths, configuration values)
- Sanitize file paths to prevent directory traversal
- Check file permissions before reading/writing
- Validate numeric ranges (e.g., test_size between 0 and 1)

#### 5.6.3 Safe File Operations
- Use pathlib.Path for cross-platform compatibility
- Check file existence before reading
- Handle permission errors gracefully
- Create directories with proper permissions

---

## 6. Success Criteria {#success-criteria}

### 6.1 Functional Success

#### 6.1.1 Model Performance
- ✅ NumPy implementation achieves **≥90% accuracy** on Iris test set
- ✅ Sklearn implementation achieves **≥90% accuracy** on Iris test set
- ✅ Prediction agreement between implementations **≥95%**
- ✅ Both models handle all 3 classes correctly
- ✅ No NaN or infinite values in predictions or probabilities

#### 6.1.2 Comparison Quality
- ✅ All comparison metrics calculated correctly (accuracy, precision, recall, F1)
- ✅ Confusion matrices generated for both implementations
- ✅ Runtime benchmarks completed (minimum 10 iterations each)
- ✅ Parameter comparisons show close agreement (MAE < 0.01 for means/variances)
- ✅ All visualizations generated successfully

#### 6.1.3 Documentation Quality
- ✅ README explains project at level accessible to 15-year-olds
- ✅ Mathematical foundations section explains Bayes' theorem clearly
- ✅ All code has comprehensive docstrings
- ✅ Inline comments explain complex logic
- ✅ Installation instructions tested and working

### 6.2 Quality Metrics

#### 6.2.1 Code Quality
- ✅ All Python files **≤150 lines** (enforced automatically)
- ✅ Every directory with Python code has `__init__.py`
- ✅ All functions have type hints
- ✅ All functions have docstrings (Google style)
- ✅ Code passes linting (flake8 or black)
- ✅ No hardcoded secrets or absolute paths

#### 6.2.2 Testing
- ✅ Minimum 3 experiment runs completed and documented
- ✅ Each run has complete results saved
- ✅ Results are reproducible with same random seed
- ✅ Unit tests written for core functions (optional but recommended)

#### 6.2.3 Documentation Completeness
- ✅ README includes all required sections (see README template)
- ✅ Code files summary table in README with line counts
- ✅ Visual results (minimum 5 graphs) embedded in README
- ✅ Mathematical explanations with examples
- ✅ Troubleshooting section with common issues

### 6.3 Performance Benchmarks

#### 6.3.1 Execution Time Targets
- ✅ Full pipeline completes in **< 15 seconds**
- ✅ NumPy training time **< 2 seconds**
- ✅ Sklearn training time **< 1 second**
- ✅ Visualization generation **< 5 seconds** total

#### 6.3.2 Accuracy Targets
- ✅ Both implementations achieve **>90% accuracy**
- ✅ Preferably achieve **>93% accuracy** (expected for Iris)
- ✅ Per-class F1-scores all **>0.90**

### 6.4 Scalability Considerations
- ✅ Code can handle datasets with more samples (e.g., 1000+) without modifications
- ✅ Can handle more features without structural changes
- ✅ Configuration file makes it easy to change parameters
- ✅ Modular design allows easy extension (add new classifiers, metrics, etc.)

---

## 7. Constraints & Assumptions {#constraints--assumptions}

### 7.1 Constraints

#### 7.1.1 Technical Constraints
- **No Sklearn in NumPy Implementation:** The manual Naive Bayes implementation must use only NumPy for core algorithm logic (sklearn allowed only for utilities like train_test_split and metrics)
- **150-Line File Limit:** Strictly enforced for all Python files
- **Python 3.10+ Required:** Cannot support older Python versions
- **UV for Virtual Environment:** Must use UV, not pip or conda
- **Time Constraint:** Project must complete full pipeline in < 15 seconds

#### 7.1.2 Resource Constraints
- **Memory:** Must run on machines with 4GB RAM minimum
- **Storage:** Complete project (including results) should be < 100 MB
- **Dependencies:** Minimize external dependencies (only necessary libraries)

#### 7.1.3 Process Constraints
- **Two-Phase Workflow:** Must complete Phase 1 (PRD + tasks.json) before Phase 2 (implementation)
- **Approval Gate:** Cannot start coding until PRD and tasks.json are approved
- **Ring Buffer Logging:** Must implement ring buffer logging system as specified in guidelines

### 7.2 Assumptions

#### 7.2.1 Data Assumptions
- **Iris.csv Availability:** Dataset is available at project start
- **Data Quality:** Iris dataset has no missing values, outliers, or errors
- **Class Balance:** All 3 classes have equal representation (50 samples each)
- **Feature Quality:** All 4 features are properly measured and meaningful for classification

#### 7.2.2 Environment Assumptions
- **Python Installation:** User has Python 3.10+ installed
- **UV Installation:** User can install UV via provided commands
- **WSL Access:** Windows users have access to WSL or can use PowerShell
- **Internet Access:** Required for initial dependency installation
- **File System Access:** User has read/write permissions in project directory

#### 7.2.3 User Assumptions
- **Basic Python Knowledge:** User understands basic Python syntax (variables, functions, loops)
- **Command Line Familiarity:** User can navigate directories and run commands
- **Mathematical Background:** User has high school level math (means, variances, probabilities)
- **Learning Attitude:** User is willing to read documentation and learn

#### 7.2.4 Implementation Assumptions
- **NumPy Correctness:** NumPy operations are numerically stable and accurate
- **Sklearn Correctness:** Sklearn's GaussianNB is a correct implementation
- **Floating Point Precision:** float64 is sufficient for Iris dataset
- **Random Seed:** Using fixed random seed (42) ensures reproducibility

---

## 8. Learning Objectives {#learning-objectives}

### 8.1 Primary Learning Objectives

This project is designed to teach and demonstrate the following concepts:

#### 8.1.1 Probability Theory and Bayes' Theorem
**Objective:** Understand how Bayes' theorem enables probabilistic classification.

**What You'll Learn:**
- **Prior Probability:** P(Class) - how common is each class?
- **Likelihood:** P(Features|Class) - what feature values are typical for each class?
- **Posterior Probability:** P(Class|Features) - given these features, what's the probability of each class?
- **Normalization:** Why posterior probabilities sum to 1
- **Maximum A Posteriori (MAP):** Choosing the class with highest posterior probability

**Real-World Connection:**
Like a doctor diagnosing a disease: they consider how common the disease is (prior), how well your symptoms match the disease (likelihood), and then calculate the probability you have the disease (posterior).

#### 8.1.2 Gaussian (Normal) Distribution
**Objective:** Understand how continuous feature values are modeled using Gaussian distributions.

**What You'll Learn:**
- **Probability Density Function (PDF):** How to calculate probability of a continuous value
- **Mean (μ):** The center of the distribution - where most values cluster
- **Variance (σ²):** The spread of the distribution - how much values vary
- **Standard Deviation (σ):** Square root of variance, easier to interpret
- **68-95-99.7 Rule:** ~68% of data within 1σ, ~95% within 2σ, ~99.7% within 3σ

**Visual Understanding:**
Imagine measuring heights of people: mean might be 170cm, and most people are within 10cm of that (standard deviation), with very few people extremely tall or short.

#### 8.1.3 "Naive" Independence Assumption
**Objective:** Understand what "naive" means in Naive Bayes and why it's useful.

**What You'll Learn:**
- **Conditional Independence:** Assuming features are independent given the class
- **Simplification:** Turns complex joint probability into product of simpler probabilities
- **Formula:** P(F₁, F₂, ..., Fₙ | C) = P(F₁|C) × P(F₂|C) × ... × P(Fₙ|C)
- **Why "Naive":** This assumption is rarely true but often works well in practice
- **When It Breaks:** Features that are correlated (e.g., height and weight)

**Intuitive Explanation:**
We pretend each feature (sepal length, sepal width, etc.) tells us about the species independently. This is "naive" because in reality they're related (larger petals often mean larger sepals), but the assumption makes calculation much simpler.

#### 8.1.4 Numerical Stability in ML
**Objective:** Understand numerical challenges in probability calculations and solutions.

**What You'll Learn:**
- **Underflow Problem:** Multiplying many small probabilities → value too small for computer to represent
- **Log Probabilities Solution:** log(a × b) = log(a) + log(b) - turns multiplication into addition
- **Epsilon Smoothing:** Adding small constant to avoid division by zero
- **Floating Point Precision:** float64 vs. float32 trade-offs

**Practical Example:**
```python
# BAD: Probability underflow
prob = 0.001 * 0.001 * 0.001 * ...  # Eventually becomes 0.0 (underflow)

# GOOD: Log probabilities
log_prob = log(0.001) + log(0.001) + log(0.001) + ...  # Stays representable
```

#### 8.1.5 Model Evaluation Methodology
**Objective:** Learn how to properly evaluate and compare machine learning models.

**What You'll Learn:**
- **Train-Test Split:** Why we need separate data for evaluation
- **Stratification:** Maintaining class balance in splits
- **Multiple Metrics:** Why accuracy alone is insufficient
  - **Accuracy:** Overall correctness
  - **Precision:** Of predicted positives, how many are actually positive?
  - **Recall:** Of actual positives, how many did we find?
  - **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Visual breakdown of correct and incorrect classifications
- **Cross-Validation:** More robust evaluation (beyond project scope, but mentioned)

**Confusion Matrix Example:**
```
              Predicted
             0    1    2
Actual   0  [10   0    0]   ← All setosa correctly predicted
         1  [ 0   9    1]   ← One versicolor misclassified as virginica
         2  [ 0   0   10]   ← All virginica correctly predicted
```

#### 8.1.6 Implementation vs. Library Code
**Objective:** Understand the value of both manual implementation and using established libraries.

**What You'll Learn:**
- **Manual Implementation Benefits:**
  - Deep understanding of algorithm internals
  - Ability to customize for specific needs
  - Better debugging skills
  - Knowledge of computational complexity

- **Library Implementation Benefits:**
  - Optimized C code (faster)
  - Well-tested (fewer bugs)
  - Maintained by experts
  - Standard interface

- **When to Use Each:**
  - Use libraries for production (reliability, performance)
  - Implement manually for learning, research, or custom requirements

**Career Insight:**
In interviews, demonstrating you can implement algorithms from scratch separates you from those who only know library APIs.

### 8.2 Secondary Learning Objectives

#### 8.2.1 Software Engineering Best Practices
- Modular code architecture (single responsibility principle)
- Type hints and documentation
- Error handling and input validation
- Configuration management
- Logging and debugging
- Version control (Git)

#### 8.2.2 Scientific Computing Skills
- NumPy vectorization for performance
- Avoiding Python loops in numerical code
- Data visualization best practices
- Reproducible experiments
- Statistical comparison methods

#### 8.2.3 Professional Documentation
- Writing for multiple audiences (beginners, experts)
- Using analogies and examples effectively
- Creating informative visualizations
- Structuring technical documents
- README as portfolio piece

### 8.3 Parameter Variations to Explore

Students should experiment with these parameters to deepen understanding:

#### 8.3.1 Train-Test Split Ratios
**Experiment:** Try 70/30, 75/25, 80/20, 90/10 splits

**What You'll Learn:**
- How split ratio affects performance variance
- Trade-off between training data and evaluation confidence
- Why 80/20 is a common default

**Expected Observation:**
Larger training sets generally improve accuracy, but with diminishing returns. Smaller test sets have higher variance in accuracy estimates.

#### 8.3.2 Random Seed Values
**Experiment:** Try different random seeds (42, 123, 999, etc.)

**What You'll Learn:**
- How data split affects results
- Concept of random variance in evaluation
- Importance of multiple runs for robust conclusions

**Expected Observation:**
Accuracy varies slightly (±2-3%) depending on which samples end up in test set. Some splits are "easier" than others.

#### 8.3.3 Epsilon (Variance Smoothing)
**Experiment:** Try epsilon values: 1e-9, 1e-6, 1e-3, 0.1

**What You'll Learn:**
- Effect of regularization on model
- Trade-off between fitting training data and generalization
- When numerical stability becomes important

**Expected Observation:**
Very small epsilon (1e-9): minimal regularization, may have numerical issues
Larger epsilon (0.1): more regularization, may underfit data

#### 8.3.4 Feature Scaling
**Experiment:** Compare results with and without standardization

**What You'll Learn:**
- When feature scaling matters (and when it doesn't)
- For Naive Bayes: per-class normalization means scaling has minimal impact
- For other algorithms: scaling is often critical

**Expected Observation:**
Naive Bayes performance should be nearly identical with/without scaling, because it estimates per-class statistics separately.

---

## 9. Mathematical Foundations {#mathematical-foundations}

### 9.1 Bayes' Theorem

#### 9.1.1 The Formula
```
P(Class | Features) = [P(Features | Class) × P(Class)] / P(Features)
```

Where:
- **P(Class | Features):** Posterior probability - what we want to find
- **P(Features | Class):** Likelihood - how likely are these features given the class
- **P(Class):** Prior probability - how common is this class
- **P(Features):** Evidence - how common are these features (used for normalization)

#### 9.1.2 Explanation for 15-Year-Olds

**Scenario:** You found a feather. Is it from a robin, eagle, or peacock?

**Prior P(Class):**
- Robins are very common in your area (prior = 0.7)
- Eagles are rare (prior = 0.1)
- Peacocks are very rare (prior = 0.2)

**Likelihood P(Feather | Bird):**
- The feather is 10 cm long and bright blue
- Robin feathers: short and brown (likelihood of blue 10cm feather = 0.01)
- Eagle feathers: long but not blue (likelihood = 0.05)
- Peacock feathers: long and can be bright blue! (likelihood = 0.9)

**Posterior P(Bird | Feather):**
Even though peacocks are rare (prior = 0.2), the feather is SO typical of peacocks (likelihood = 0.9) that peacock becomes the most likely source.

**Calculation:**
```
P(Peacock | Blue feather) ∝ 0.9 × 0.2 = 0.18
P(Eagle | Blue feather) ∝ 0.05 × 0.1 = 0.005
P(Robin | Blue feather) ∝ 0.01 × 0.7 = 0.007

After normalization:
P(Peacock | Blue feather) = 0.18 / (0.18 + 0.005 + 0.007) ≈ 0.94 (94%)
```

**Conclusion:** Despite peacocks being rare, the evidence (blue, long feather) strongly suggests peacock.

### 9.2 Gaussian Naive Bayes

#### 9.2.1 The Full Algorithm

**Training Phase:**
```
For each class c ∈ {0, 1, 2}:
    Prior: P(c) = count(samples in class c) / total_samples

    For each feature f ∈ {0, 1, 2, 3}:
        μ[c][f] = mean(feature f values for class c samples)
        σ²[c][f] = variance(feature f values for class c samples) + epsilon
```

**Prediction Phase:**
```
For each test sample x:
    For each class c ∈ {0, 1, 2}:
        log_posterior[c] = log(P(c))

        For each feature i ∈ {0, 1, 2, 3}:
            # Gaussian PDF
            likelihood = (1/√(2π σ²[c][i])) × exp(-(x[i] - μ[c][i])² / (2σ²[c][i]))
            log_posterior[c] += log(likelihood)

    prediction = argmax(log_posterior)
```

#### 9.2.2 Why Gaussian?

**What is Gaussian Distribution?**

The Gaussian (normal) distribution is a bell-shaped curve describing how continuous values are distributed. It's defined by:
- **Mean (μ):** Center of the bell curve
- **Variance (σ²):** Width of the bell curve
- **Formula:** `P(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))`

**Why Gaussian for Iris?**

Iris features (petal length, sepal width, etc.) are continuous measurements that tend to cluster around typical values for each species:
- Setosa: typically small petals (mean ≈ 1.5 cm, variance small)
- Versicolor: medium petals (mean ≈ 4.3 cm)
- Virginica: large petals (mean ≈ 5.6 cm)

Gaussian distribution models this clustering well!

#### 9.2.3 The "Naive" Assumption

**What Does "Naive" Mean?**

We assume features are **conditionally independent** given the class:
```
P(F₁, F₂, F₃, F₄ | Class) = P(F₁|Class) × P(F₂|Class) × P(F₃|Class) × P(F₄|Class)
```

**Reality Check:**

In reality, features ARE correlated:
- Larger petals often mean larger sepals
- Petal length and petal width are correlated
- So the independence assumption is **false** ("naive")

**Why It Works Anyway:**

Despite being "naive," the algorithm works well because:
1. We're not predicting exact probabilities, just ranking classes
2. Errors in probability estimates often cancel out
3. The relative ordering of classes is often preserved
4. Simple model = less overfitting = better generalization

**Mathematical Justification:**

Even if true probabilities are:
```
P(C₁|Features) = 0.8  (actual)
P(C₂|Features) = 0.2  (actual)
```

Our naive estimates might be:
```
P̂(C₁|Features) = 0.6  (estimated)
P̂(C₂|Features) = 0.1  (estimated)
```

The estimates are wrong, but the **ranking is correct** (C₁ > C₂), so we still predict the right class!

### 9.3 Numerical Stability: Log Probabilities

#### 9.3.1 The Underflow Problem

**Problem:**

When multiplying many small probabilities, the result can become too small for the computer to represent:
```
P(Features | Class) = P(F₁|C) × P(F₂|C) × P(F₃|C) × P(F₄|C)
                    ≈ 0.01 × 0.05 × 0.02 × 0.03
                    = 0.00000003
```

For longer feature vectors, this can become `0.0` (underflow).

#### 9.3.2 Log Probability Solution

**Key Identity:** `log(a × b) = log(a) + log(b)`

Multiplication becomes addition:
```python
# Instead of:
prob = p1 * p2 * p3 * p4  # Can underflow

# Use:
log_prob = log(p1) + log(p2) + log(p3) + log(p4)  # Safe!
```

**Benefits:**
- No underflow (log of small number is negative, not zero)
- Numerically stable
- Addition is faster than multiplication
- Easier to debug (can check individual log-likelihood terms)

**Converting Back:**
```python
# If needed, convert back to probability:
prob = np.exp(log_prob)

# But for classification, we only need argmax, so we can compare log probabilities directly:
prediction = np.argmax(log_posteriors)  # No need to exponentiate!
```

### 9.4 Example Calculation Walkthrough

Let's classify one Iris sample step-by-step.

#### Sample Input
```
Features: [5.9, 3.0, 5.1, 1.8]
True Label: Class 2 (Virginica)
```

#### Step 1: Priors (from training)
```
P(Class 0) = 40/120 = 0.333
P(Class 1) = 40/120 = 0.333
P(Class 2) = 40/120 = 0.333
```

#### Step 2: Learned Parameters (example values)
```
Class 0 (Setosa):
  μ = [5.0, 3.4, 1.5, 0.2]
  σ² = [0.12, 0.14, 0.03, 0.01]

Class 1 (Versicolor):
  μ = [5.9, 2.8, 4.3, 1.3]
  σ² = [0.27, 0.10, 0.22, 0.04]

Class 2 (Virginica):
  μ = [6.6, 3.0, 5.6, 2.0]
  σ² = [0.40, 0.10, 0.30, 0.08]
```

#### Step 3: Calculate Log Likelihoods

**For Class 0:**
```
Feature 0 (Sepal Length = 5.9):
  μ₀ = 5.0, σ₀² = 0.12
  gaussian_pdf = (1/√(2π×0.12)) × exp(-(5.9-5.0)²/(2×0.12))
  ≈ 0.0023
  log_likelihood ≈ -6.07

[Calculate for features 1, 2, 3...]
Total log_likelihood for Class 0 ≈ -45.23
```

**For Class 1:**
```
Feature 0 (Sepal Length = 5.9):
  μ₁ = 5.9, σ₁² = 0.27
  gaussian_pdf = (1/√(2π×0.27)) × exp(-(5.9-5.9)²/(2×0.27))
  ≈ 0.77  (Much higher! Feature matches mean perfectly)
  log_likelihood ≈ -0.26

[Calculate for features 1, 2, 3...]
Total log_likelihood for Class 1 ≈ -12.45
```

**For Class 2:**
```
Feature 0 (Sepal Length = 5.9):
  μ₂ = 6.6, σ₂² = 0.40
  gaussian_pdf ≈ 0.45
  log_likelihood ≈ -0.80

[Calculate for features 1, 2, 3...]
Total log_likelihood for Class 2 ≈ -8.91  ← HIGHEST
```

#### Step 4: Log Posteriors
```
log P(C₀|x) = log(0.333) + (-45.23) = -1.10 + (-45.23) = -46.33
log P(C₁|x) = log(0.333) + (-12.45) = -1.10 + (-12.45) = -13.55
log P(C₂|x) = log(0.333) + (-8.91) = -1.10 + (-8.91) = -10.01  ← HIGHEST
```

#### Step 5: Prediction
```
prediction = argmax([-46.33, -13.55, -10.01]) = 2 (Virginica)
```

**Result:** Correctly classified as Virginica! ✅

---

## 10. Implementation Comparison Strategy {#implementation-comparison-strategy}

### 10.1 Fair Comparison Principles

#### 10.1.1 Identical Data Splits
**Requirement:** Both implementations must use **exactly the same** train/test split.

**Implementation:**
```python
# Split once, use for both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# NumPy model
model_numpy = GaussianNaiveBayesNumPy()
model_numpy.fit(X_train, y_train)
pred_numpy = model_numpy.predict(X_test)

# Sklearn model (SAME split)
model_sklearn = GaussianNaiveBayesSKlearn()
model_sklearn.fit(X_train, y_train)
pred_sklearn = model_sklearn.predict(X_test)
```

#### 10.1.2 Matched Hyperparameters
**Requirement:** Use equivalent smoothing parameters.

- NumPy: `epsilon = 1e-9`
- Sklearn: `var_smoothing = 1e-9`

Both prevent division by zero by adding small constant to variance.

#### 10.1.3 Consistent Evaluation
**Requirement:** Calculate all metrics using the same methodology.

Use sklearn's metric functions for both:
```python
# Both use sklearn.metrics for consistency
acc_numpy = accuracy_score(y_test, pred_numpy)
acc_sklearn = accuracy_score(y_test, pred_sklearn)
```

### 10.2 Comparison Dimensions

#### 10.2.1 Accuracy Metrics
Compare:
- Overall accuracy
- Per-class precision
- Per-class recall
- Per-class F1-score
- Macro-averaged metrics
- Weighted-averaged metrics

**Expected Outcome:** Very similar metrics (within 1-2%)

#### 10.2.2 Prediction Agreement
**Metric:** Percentage of test samples where both models predict the same class

```python
agreement = np.mean(pred_numpy == pred_sklearn) * 100
print(f"Agreement: {agreement:.2f}%")
```

**Expected Outcome:** ≥95% agreement

**Analysis of Disagreements:**
- Which samples cause disagreement?
- Are these samples near decision boundaries?
- What are the confidence levels for disagreements?

#### 10.2.3 Parameter Comparison
Compare learned parameters:
- **Priors:** Should be identical (just sample counts)
- **Means:** Should match closely (same data, same calculation)
- **Variances:** May differ slightly due to different variance estimators

**Metrics:**
```python
mae_means = np.mean(np.abs(means_numpy - means_sklearn))
mae_vars = np.mean(np.abs(vars_numpy - vars_sklearn))
```

**Expected Outcome:**
- MAE for means < 0.01
- MAE for variances < 0.01

#### 10.2.4 Runtime Performance
Benchmark:
- Training time (fit)
- Prediction time (predict)
- Total time (fit + predict)

**Method:**
```python
import time

# Multiple runs for statistical significance
times = []
for _ in range(10):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    end = time.perf_counter()
    times.append(end - start)

mean_time = np.mean(times)
std_time = np.std(times)
```

**Expected Outcome:**
- Sklearn faster (optimized C code)
- NumPy reasonable (within 2-5x of sklearn)
- Both fast enough for Iris dataset (< 2 seconds)

#### 10.2.5 Memory Usage
Compare peak memory usage during training/prediction.

**Method:**
```python
import tracemalloc

tracemalloc.start()
model.fit(X_train, y_train)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
```

**Expected Outcome:**
- Both use minimal memory (< 10 MB for Iris)
- Sklearn may use slightly more (object overhead)

### 10.3 Presenting Results

#### 10.3.1 Side-by-Side Comparison Table
```markdown
| Metric              | NumPy Implementation | Sklearn Implementation | Difference |
|---------------------|---------------------|------------------------|------------|
| Accuracy            | 93.33%              | 93.33%                 | 0.00%      |
| Macro Avg Precision | 93.89%              | 93.89%                 | 0.00%      |
| Macro Avg Recall    | 93.33%              | 93.33%                 | 0.00%      |
| Macro Avg F1        | 93.33%              | 93.33%                 | 0.00%      |
| Training Time       | 1.23 ± 0.05 ms      | 0.78 ± 0.03 ms         | 1.58x slower |
| Prediction Time     | 0.45 ± 0.02 ms      | 0.28 ± 0.01 ms         | 1.61x slower |
| Agreement Rate      | -                   | -                      | 100%       |
```

#### 10.3.2 Visual Comparison
- **Graph 1:** Bar chart of metric comparison
- **Graph 2:** Side-by-side confusion matrices
- **Graph 3:** Runtime comparison bar chart
- **Graph 4:** Parameter scatter plot (NumPy vs sklearn)

#### 10.3.3 Analysis Section
Interpret the comparison:
- **Why metrics are similar:** Same algorithm, same data, same parameters
- **Why sklearn is faster:** Compiled C code, optimized implementation
- **Why there are any differences:** Numerical precision, variance estimator details, implementation-specific optimizations
- **Conclusion:** NumPy implementation is correct (validated against sklearn)

---

## 11. Deliverables {#deliverables}

### 11.1 Code Deliverables

#### 11.1.1 Root Directory
- ✅ `README.md` - Comprehensive project documentation (serving dual purpose: docs + learning guide)
- ✅ `main.py` - Single entry point to run the entire pipeline
- ✅ `requirements.txt` - Exact dependency versions (generated via `uv pip freeze`)
- ✅ `.gitignore` - Git ignore file (secrets, cache, venv)
- ✅ `.env.example` - Template for environment variables (if needed)

#### 11.1.2 Virtual Environment Indicator
- ✅ `venv/.gitkeep` - Virtual environment indicator with setup instructions

#### 11.1.3 Source Code (`src/`)
- ✅ `src/__init__.py` - Package marker
- ✅ `src/naive_bayes_manual.py` - NumPy Gaussian Naive Bayes implementation (≤150 lines)
- ✅ `src/naive_bayes_sklearn.py` - Sklearn wrapper implementation (≤150 lines)
- ✅ `src/data_loader.py` - Data loading and splitting (≤150 lines)
- ✅ `src/evaluator.py` - Metrics calculation and comparison (≤150 lines)
- ✅ `src/visualizer.py` - Visualization generation (≤150 lines)
- ✅ `src/utils/__init__.py` - Utilities package marker
- ✅ `src/utils/logger.py` - Ring buffer logging implementation (≤150 lines)
- ✅ `src/utils/validators.py` - Input validation functions (≤150 lines)
- ✅ `src/utils/helpers.py` - Utility helper functions (≤150 lines)
- ✅ `src/utils/paths.py` - Path management utilities (≤150 lines)

#### 11.1.4 Documentation (`docs/`)
- ✅ `docs/PRD.md` - This Product Requirements Document
- ✅ `docs/tasks.json` - Detailed task breakdown with dependencies

#### 11.1.5 Configuration (`config/`)
- ✅ `config/settings.yaml` - Main configuration file
- ✅ `config/log_config.json` - Ring buffer logging configuration
- ✅ `config/experiments/baseline.yaml` - Baseline experiment configuration
- ✅ `config/experiments/different_split.yaml` - Alternative split configuration
- ✅ `config/experiments/with_scaling.yaml` - Feature scaling experiment configuration

#### 11.1.6 Tests (`tests/`) - Optional but Recommended
- ✅ `tests/__init__.py` - Test package marker
- ✅ `tests/test_naive_bayes_manual.py` - Unit tests for NumPy implementation
- ✅ `tests/test_naive_bayes_sklearn.py` - Unit tests for sklearn wrapper
- ✅ `tests/test_data_loader.py` - Unit tests for data loading

### 11.2 Data Deliverables

#### 11.2.1 Dataset (`data/`)
- ✅ `data/Iris.csv` - Iris flower dataset (150 samples)
- ✅ `data/README.md` - Data source information and description

### 11.3 Results Deliverables

#### 11.3.1 Experiment Results (`results/examples/`)
- ✅ `results/examples/run_1/` - Baseline experiment
  - `config.yaml` - Configuration used
  - `metrics.json` - All calculated metrics
  - `predictions.csv` - Model predictions
  - `visualizations/` - Run-specific graphs
- ✅ `results/examples/run_2/` - Second experiment (different split)
- ✅ `results/examples/run_3/` - Third experiment (with scaling)

#### 11.3.2 Visualizations (`results/graphs/`)
- ✅ `results/graphs/feature_distributions_numpy.png` - Feature histograms (NumPy run)
- ✅ `results/graphs/feature_distributions_sklearn.png` - Feature histograms (sklearn run)
- ✅ `results/graphs/confusion_matrix_comparison.png` - Side-by-side confusion matrices
- ✅ `results/graphs/metric_comparison.png` - Bar chart comparing metrics
- ✅ `results/graphs/runtime_comparison.png` - Bar chart comparing runtimes
- ✅ `results/graphs/parameter_scatter.png` - Scatter plot of learned parameters

#### 11.3.3 Summary Tables (`results/tables/`)
- ✅ `results/tables/metrics_summary.csv` - All metrics across all runs
- ✅ `results/tables/runtime_summary.csv` - Runtime benchmarks
- ✅ `results/tables/parameter_comparison.csv` - Parameter comparison

### 11.4 Logging Deliverables (`logs/`)
- ✅ `logs/config/log_config.json` - Ring buffer logging configuration
- ✅ `logs/.gitkeep` - Keep folder in git (actual logs ignored)
- ✅ Log files generated during runs (not committed to git)

### 11.5 Expected Outputs After Running

After running `python main.py`, the following outputs should be generated:

#### Console Output
```
==============================================================
🚀 Starting L21 - Iris Naive Bayes Classification
==============================================================

📁 Step 1/10: Loading configuration...
✅ Configuration loaded: config/settings.yaml

📁 Step 2/10: Loading and splitting data...
✅ Loaded 150 samples with 4 features
✅ Split: 120 training, 30 testing (stratified)

🧮 Step 3/10: Training NumPy Naive Bayes...
   Learned Parameters:
     Class 0 (Setosa):     Prior=0.333, μ=[5.01, 3.42, ...], σ²=[0.12, 0.14, ...]
     Class 1 (Versicolor): Prior=0.333, μ=[5.94, 2.77, ...], σ²=[0.27, 0.10, ...]
     Class 2 (Virginica):  Prior=0.333, μ=[6.59, 2.97, ...], σ²=[0.40, 0.10, ...]
✅ Training complete in 1.23 ms

🧮 Step 4/10: Training Sklearn Naive Bayes...
✅ Training complete in 0.78 ms

🎯 Step 5/10: Generating predictions...
✅ NumPy predictions: 28/30 correct (93.33%)
✅ Sklearn predictions: 28/30 correct (93.33%)
✅ Prediction agreement: 100.00% (30/30)

📊 Step 6/10: Calculating metrics...
✅ Metrics calculated for both implementations

⏱️  Step 7/10: Running performance benchmarks...
✅ Benchmarks complete (10 iterations each)

📈 Step 8/10: Generating visualizations...
✅ Saved: results/graphs/feature_distributions_numpy.png
✅ Saved: results/graphs/confusion_matrix_comparison.png
✅ Saved: results/graphs/metric_comparison.png
✅ Saved: results/graphs/runtime_comparison.png
✅ Saved: results/graphs/parameter_scatter.png

💾 Step 9/10: Saving results...
✅ Saved: results/examples/run_1/metrics.json
✅ Saved: results/examples/run_1/predictions.csv
✅ Saved: results/tables/metrics_summary.csv

📊 Step 10/10: Printing summary...

==============================================================
📊 RESULTS SUMMARY
==============================================================

Implementation Comparison:
                    NumPy    Sklearn  Difference
Accuracy            93.33%   93.33%   0.00%
Macro Avg Precision 93.89%   93.89%   0.00%
Macro Avg Recall    93.33%   93.33%   0.00%
Macro Avg F1        93.33%   93.33%   0.00%

Runtime Performance:
                NumPy          Sklearn        Speedup
Training        1.23 ± 0.05 ms 0.78 ± 0.03 ms 1.58x
Prediction      0.45 ± 0.02 ms 0.28 ± 0.01 ms 1.61x

Agreement Rate: 100.00%

==============================================================
✅ Pipeline completed successfully!
📂 Check results/ directory for all outputs
==============================================================

📊 Log Status:
  Files: 1/5
  Total Lines: 234
  Total Size: 12.45 KB
```

---

## 12. Risk Management {#risk-management}

### 12.1 Technical Risks

#### Risk 1: Numerical Underflow in NumPy Implementation
**Probability:** Medium
**Impact:** High (incorrect predictions)
**Mitigation:**
- Use log probabilities throughout
- Add epsilon smoothing to variances
- Test with edge cases (very small/large values)
- Validate against sklearn results

#### Risk 2: File Size Exceeding 150 Lines
**Probability:** High (without discipline)
**Impact:** Medium (violates guidelines)
**Mitigation:**
- Plan module boundaries before coding
- Enforce with pre-commit hook
- Use code splitting strategically
- Prioritize single responsibility principle

#### Risk 3: Dependency Version Conflicts
**Probability:** Low
**Impact:** High (project won't run)
**Mitigation:**
- Use UV for dependency management
- Pin exact versions in requirements.txt
- Test in fresh virtual environment
- Document Python version requirement clearly

#### Risk 4: Performance Below Expectations
**Probability:** Low
**Impact:** Medium (slow execution)
**Mitigation:**
- Use NumPy vectorization (no Python loops)
- Profile code to identify bottlenecks
- Optimize critical sections if needed
- Iris dataset is small enough for reasonable performance

#### Risk 5: Visualization Generation Fails
**Probability:** Low
**Impact:** Medium (missing deliverables)
**Mitigation:**
- Test visualization code separately
- Handle matplotlib backend issues
- Provide clear error messages
- Fallback to text-based outputs if necessary

### 12.2 Process Risks

#### Risk 6: Skipping Phase 1 Approval
**Probability:** Low (explicitly warned against)
**Impact:** Critical (wasted implementation effort)
**Mitigation:**
- Clear instructions in guidelines
- Explicit approval checkpoint in tasks.json
- Reminder in main.py template
- PRD and tasks.json review before proceeding

#### Risk 7: Inadequate Documentation
**Probability:** Medium
**Impact:** High (fails learning objective)
**Mitigation:**
- Allocate sufficient time for documentation
- Use templates for consistency
- Review against checklist
- Get peer review if possible

#### Risk 8: Results Not Reproducible
**Probability:** Medium
**Impact:** High (credibility issue)
**Mitigation:**
- Use fixed random seeds everywhere
- Version-lock all dependencies
- Document environment setup meticulously
- Test reproducibility in fresh environment

### 12.3 Educational Risks

#### Risk 9: Code Too Complex for Target Audience
**Probability:** Medium
**Impact:** High (defeats educational purpose)
**Mitigation:**
- Extensive code comments explaining WHY
- Use simple variable names
- Break complex logic into smaller functions
- Provide analogies and examples

#### Risk 10: Missing Learning Objectives
**Probability:** Low
**Impact:** Medium (reduced educational value)
**Mitigation:**
- Explicit learning objectives in PRD
- Design experiments to teach key concepts
- Include "What You'll Learn" sections
- Peer review from educational perspective

---

## 13. Approval & Sign-off

**Status:** ⏳ Awaiting Approval

This PRD must be reviewed and approved before proceeding to Phase 2 (Implementation).

**Review Checklist:**
- ✅ All required sections present and complete
- ✅ Success criteria clearly defined
- ✅ Learning objectives align with course goals
- ✅ Technical requirements are achievable
- ✅ Comparison methodology is fair and scientific
- ✅ Documentation standards meet guidelines
- ✅ File structure follows PROJECT_GUIDELINES.md
- ✅ Deliverables list is comprehensive
- ✅ Risk mitigation strategies are sound

**Reviewer:** [To be filled]
**Approval Date:** [To be filled]
**Notes:** [To be filled]

---

**END OF PRD**

*This document serves as the complete specification for the Iris Naive Bayes classification project. No implementation should begin until this PRD and the accompanying tasks.json are reviewed and approved.*
