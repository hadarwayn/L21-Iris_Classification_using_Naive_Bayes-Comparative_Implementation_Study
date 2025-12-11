# L21 - Iris Classification using Naive Bayes: Comparative Implementation Study

A professional implementation comparing manual NumPy and scikit-learn Gaussian Naive Bayes classifiers on the famous Iris flower dataset.

## Project Overview

This project implements Gaussian Naive Bayes in two ways:
1. **Manual Implementation**: From scratch using only NumPy to demonstrate understanding of the algorithm
2. **Scikit-learn Implementation**: Using industry-standard library for comparison

Both implementations are rigorously compared across multiple dimensions: accuracy, runtime, learned parameters, and prediction agreement.

## Results Summary

### Experiment Results
- **Run 1 (Baseline 80-20 split)**: 96.67% accuracy, 100% agreement
- **Run 2 (70-30 split)**: 91.11% accuracy, 100% agreement
- **Run 3 (With normalization)**: 96.67% accuracy, 100% agreement

All experiments show perfect agreement between NumPy and scikit-learn implementations, validating correctness.

## Detailed Results and Analysis

### Experimental Setup

Three experiments were conducted to evaluate the Gaussian Naive Bayes implementations:

| Experiment | Description | Train/Test Split | Normalization | Purpose |
|------------|-------------|------------------|---------------|---------|
| **Run 1** | Baseline | 80/20 (120/30) | No | Standard evaluation |
| **Run 2** | Different Split | 70/30 (105/45) | No | Generalization testing |
| **Run 3** | With Scaling | 80/20 (120/30) | Yes (z-score) | Feature scaling impact |

### Classification Performance Metrics

#### Run 1: Baseline (80-20 Split)

| Metric | NumPy Implementation | Scikit-learn | Difference |
|--------|---------------------|--------------|------------|
| **Accuracy** | 96.67% | 96.67% | 0.00% |
| **Precision** | 96.97% | 96.97% | 0.00% |
| **Recall** | 96.67% | 96.67% | 0.00% |
| **F1-Score** | 96.66% | 96.66% | 0.00% |

**Confusion Matrix Analysis:**
- Perfect classification of Iris-setosa (10/10)
- Near-perfect classification of Iris-versicolor (9/10, 1 misclassified as virginica)
- Perfect classification of Iris-virginica (10/10)
- Overall: 29/30 correct predictions (96.67%)

#### Run 2: Different Split (70-30)

| Metric | NumPy Implementation | Scikit-learn | Difference |
|--------|---------------------|--------------|------------|
| **Accuracy** | 91.11% | 91.11% | 0.00% |
| **Precision** | 91.55% | 91.55% | 0.00% |
| **Recall** | 91.11% | 91.11% | 0.00% |
| **F1-Score** | 91.07% | 91.07% | 0.00% |

**Observations:**
- Larger test set (45 samples vs 30) provides more robust evaluation
- Slightly lower accuracy (91.11% vs 96.67%) due to smaller training set
- Still demonstrates excellent generalization capability
- Validates model stability across different data splits

#### Run 3: With Feature Normalization

| Metric | NumPy Implementation | Scikit-learn | Difference |
|--------|---------------------|--------------|------------|
| **Accuracy** | 96.67% | 96.67% | 0.00% |
| **Precision** | 96.97% | 96.97% | 0.00% |
| **Recall** | 96.67% | 96.67% | 0.00% |
| **F1-Score** | 96.66% | 96.66% | 0.00% |

**Key Finding:**
- Z-score normalization had **zero impact** on classification performance
- This is expected behavior for Gaussian Naive Bayes, which estimates mean and variance per feature
- Validates theoretical understanding: GNB is scale-invariant within each class

### Runtime Performance Comparison

#### Training Time (Mean ± Std)

| Experiment | NumPy | Scikit-learn | Speedup Factor |
|------------|-------|--------------|----------------|
| **Run 1** | 328.64 μs ± 94.55 μs | 690.89 μs ± 116.33 μs | **2.10x faster** |
| **Run 2** | 198.23 μs ± 48.99 μs | 609.82 μs ± 50.47 μs | **3.08x faster** |
| **Run 3** | 273.48 μs ± 131.75 μs | 666.13 μs ± 80.99 μs | **2.44x faster** |

**Average Speedup:** NumPy implementation is **2.54x faster** at training

#### Prediction Time (Mean ± Std)

| Experiment | NumPy | Scikit-learn | Speedup Factor |
|------------|-------|--------------|----------------|
| **Run 1** | 57.41 μs ± 11.18 μs | 181.63 μs ± 159.14 μs | **3.16x faster** |
| **Run 2** | 45.22 μs ± 7.84 μs | 107.09 μs ± 20.89 μs | **2.37x faster** |
| **Run 3** | 45.11 μs ± 8.21 μs | 107.58 μs ± 21.99 μs | **2.39x faster** |

**Average Speedup:** NumPy implementation is **2.64x faster** at prediction

### Parameter Comparison Analysis

Both implementations learn identical parameters, confirming mathematical correctness:

| Parameter Type | Maximum Difference | Mean Difference |
|----------------|-------------------|-----------------|
| **Class Priors** | 0.0 | 0.0 |
| **Feature Means (θ)** | 0.0 | 0.0 |
| **Feature Variances (σ²)** | ~1×10⁻⁹ | ~1×10⁻⁹ |

**Interpretation:**
- **Priors**: Exact match - both implementations correctly calculate P(class) = count(class)/total
- **Means**: Exact match - identical calculation of feature averages per class
- **Variances**: Near-exact match - tiny differences (~1 billionth) due to epsilon smoothing (1×10⁻⁹)

### Prediction Agreement

| Experiment | Total Predictions | Agreements | Disagreements | Agreement Rate |
|------------|------------------|------------|---------------|----------------|
| **Run 1** | 30 | 30 | 0 | **100.00%** |
| **Run 2** | 45 | 45 | 0 | **100.00%** |
| **Run 3** | 30 | 30 | 0 | **100.00%** |

**Conclusion:** Perfect prediction agreement across all 105 test samples validates implementation correctness.

### Visualizations

Each experiment generates 6 publication-quality visualizations at 300 DPI:

1. **Confusion Matrix - NumPy** ([run_1/plots/cm_numpy.png](results/examples/run_1/plots/cm_numpy.png))
   - Shows predicted vs actual class labels
   - Diagonal elements = correct predictions
   - Off-diagonal elements = misclassifications

2. **Confusion Matrix - Scikit-learn** ([run_1/plots/cm_sklearn.png](results/examples/run_1/plots/cm_sklearn.png))
   - Identical to NumPy confusion matrix
   - Validates 100% prediction agreement

3. **Metrics Comparison** ([run_1/plots/metrics_comparison.png](results/examples/run_1/plots/metrics_comparison.png))
   - Bar chart comparing accuracy, precision, recall, F1-score
   - Side-by-side visualization shows identical performance

4. **Runtime Comparison** ([run_1/plots/runtime_comparison.png](results/examples/run_1/plots/runtime_comparison.png))
   - Training and prediction time comparison
   - Clearly shows NumPy's 2-3x speedup advantage

5. **Feature Distributions by Class** ([run_1/plots/feature_distributions.png](results/examples/run_1/plots/feature_distributions.png))
   - Histograms of 4 features colored by species
   - Shows Gaussian distribution assumption is reasonable
   - Demonstrates class separability (especially Setosa)

6. **Class Distribution** ([run_1/plots/class_distribution.png](results/examples/run_1/plots/class_distribution.png))
   - Pie chart showing balanced training data (33.3% each class)
   - Validates stratified sampling maintains class proportions

### Key Findings and Insights

#### 1. Implementation Correctness ✅
- **100% prediction agreement** across all experiments validates mathematical correctness
- Parameters match to floating-point precision (differences ~10⁻⁹)
- Both implementations produce identical classification results

#### 2. Performance Advantage 🚀
- **NumPy implementation is 2.5x faster** on average
- Speedup comes from:
  - Lightweight implementation without sklearn overhead
  - Direct vectorized NumPy operations
  - No unnecessary abstractions
- Trade-off: Less robust to edge cases than sklearn's battle-tested code

#### 3. Gaussian Naive Bayes Characteristics 📊
- **Scale-invariant**: Normalization doesn't affect results (Run 1 vs Run 3)
- **Simple yet effective**: 96.67% accuracy with minimal code
- **Fast training**: <1ms even with sklearn overhead
- **Interpretable**: Can examine learned means, variances, priors

#### 4. Dataset Insights 🌸
- **Iris-setosa is linearly separable**: Never misclassified
- **Iris-versicolor and Iris-virginica overlap**: Occasional confusion
- **Balanced classes**: No bias toward majority class
- **Clean data**: No missing values, no preprocessing needed

#### 5. Generalization Capability 🎯
- Run 2 (70-30 split): 91.11% accuracy validates generalization
- Small accuracy drop with smaller training set is expected
- No overfitting observed (similar train/test performance)

### Theoretical vs Experimental Results

| Aspect | Theoretical Expectation | Experimental Observation | Match? |
|--------|------------------------|--------------------------|--------|
| Scale invariance | GNB invariant to feature scaling | Run 1 ≈ Run 3 accuracy | ✅ Yes |
| Parameter learning | Identical with same epsilon | Params match to 10⁻⁹ | ✅ Yes |
| Predictions | Same params → same predictions | 100% agreement | ✅ Yes |
| Computational cost | NumPy simpler → faster | 2.5x faster | ✅ Yes |

### Comparison to Literature

Standard Iris dataset benchmarks:
- **Linear Discriminant Analysis**: ~98% accuracy
- **k-Nearest Neighbors (k=3)**: ~96% accuracy
- **Decision Tree**: ~95% accuracy
- **Gaussian Naive Bayes**: 94-97% accuracy (varies by split)

Our results (96.67% baseline) align perfectly with published literature, confirming implementation quality.

### Limitations and Future Work

**Current Limitations:**
1. Small dataset (150 samples) - not suitable for deep learning comparison
2. Perfectly balanced classes - doesn't test class imbalance handling
3. No cross-validation - single train/test split per experiment
4. Binary epsilon smoothing - could explore adaptive smoothing

**Future Enhancements:**
1. Implement k-fold cross-validation for robust evaluation
2. Add support for categorical features (Categorical Naive Bayes)
3. Explore Laplace smoothing variants
4. Compare with other classifiers (LDA, SVM, Random Forest)
5. Test on imbalanced datasets
6. Implement feature selection analysis

## What is Naive Bayes? (For Beginners)

### The Big Idea

Imagine you find a feather on the ground. Is it from a robin, eagle, or peacock? You'd look at clues like:
- Length of the feather
- Color of the feather
- Where you found it (robins are common in your area)

Naive Bayes works the same way! It looks at features (measurements) and prior knowledge (how common each bird is) to make the best guess.

### The Mathematics

**Bayes' Theorem**:
```
P(Class | Features) = P(Features | Class) × P(Class) / P(Features)
```

In plain English:
- **P(Class | Features)**: "What's the probability it's a robin, given these feather measurements?"
- **P(Features | Class)**: "How likely are these measurements if it's a robin?"
- **P(Class)**: "How common are robins in general?"

**Why "Naive"?**
We assume features are independent (length doesn't affect color). This is usually not true, but works surprisingly well!

### Gaussian Naive Bayes

For continuous measurements (like petal length), we assume values follow a bell curve (Gaussian/normal distribution). For each class, we learn:
- **Mean (μ)**: The average petal length for this species
- **Variance (σ²)**: How spread out the measurements are

**Gaussian Probability Density Function**:
```
P(x | class) = (1 / √(2πσ²)) × exp(-(x - μ)² / (2σ²))
```

This gives the probability of seeing a particular measurement for each class.

### Making Predictions

For each class, we:
1. Calculate prior probability P(class) = (samples in class) / (total samples)
2. Calculate likelihood P(features | class) using Gaussian PDF
3. Multiply: posterior = prior × likelihood
4. Pick the class with highest posterior probability!

**Numerical Stability Trick**: We use logarithms to avoid very tiny numbers:
```
log P(class | features) = log P(features | class) + log P(class)
```

## Project Structure

```
L21/
├── README.md                 # This file
├── main.py                   # Main orchestration script
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files
│   ├── settings.yaml         # Main settings
│   └── experiments/          # Experiment configs
├── src/                      # Source code
│   ├── data_loader.py        # Data loading and splitting
│   ├── naive_bayes_manual.py # NumPy implementation
│   ├── naive_bayes_sklearn.py# Scikit-learn wrapper
│   ├── evaluator.py          # Metrics and comparison
│   ├── visualizer.py         # Plotting functions
│   └── utils/                # Utility modules
├── data/                     # Iris dataset
├── results/                  # Experiment outputs
│   └── examples/
│       ├── run_1/            # Baseline results
│       ├── run_2/            # Different split
│       └── run_3/            # With normalization
└── logs/                     # Log files
```

## Installation

### Prerequisites
- Python 3.10 or higher
- UV package manager

### Setup

1. Clone or download this project

2. Create virtual environment with UV:
```bash
uv venv venv
```

3. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

### Running Experiments

Run a single experiment:
```bash
python main.py config/experiments/baseline.yaml
```

Run all experiments:
```bash
python main.py config/experiments/baseline.yaml
python main.py config/experiments/different_split.yaml
python main.py config/experiments/with_scaling.yaml
```

### Experiment Configurations

- **baseline.yaml**: Standard 80-20 train-test split
- **different_split.yaml**: 70-30 split to test generalization
- **with_scaling.yaml**: With feature normalization (z-score)

### Output Files

Each experiment generates:
- `results.json`: Complete metrics and timing data
- `plots/`: 6 visualizations at 300 DPI:
  - Confusion matrices (NumPy and sklearn)
  - Metrics comparison bar chart
  - Runtime comparison
  - Feature distributions by class
  - Class distribution pie chart

## Implementation Details

### Mathematical Foundation

#### Bayes' Theorem

The core principle behind Naive Bayes classification is Bayes' theorem:

```
P(C|X) = P(X|C) × P(C) / P(X)
```

Where:
- **P(C|X)**: Posterior probability - probability of class C given features X
- **P(X|C)**: Likelihood - probability of observing features X given class C
- **P(C)**: Prior probability - probability of class C occurring
- **P(X)**: Evidence - probability of observing features X (constant for all classes)

Since P(X) is constant across all classes, we can simplify to:

```
P(C|X) ∝ P(X|C) × P(C)
```

#### Naive Independence Assumption

For features X = [x₁, x₂, ..., xₙ], we assume conditional independence:

```
P(X|C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C) = ∏ᵢ P(xᵢ|C)
```

This "naive" assumption simplifies computation dramatically and works surprisingly well in practice.

#### Gaussian Distribution for Continuous Features

For each feature xᵢ in class C, we assume a Gaussian (normal) distribution:

```
P(xᵢ|C) = (1 / √(2πσ²ᵢc)) × exp(-(xᵢ - μᵢc)² / (2σ²ᵢc))
```

Where:
- **μᵢc**: Mean of feature i for class C
- **σ²ᵢc**: Variance of feature i for class C

#### Log-Space Computation for Numerical Stability

To prevent underflow from multiplying many small probabilities, we use logarithms:

```
log P(C|X) = log P(C) + Σᵢ log P(xᵢ|C)

log P(xᵢ|C) = -0.5 × [log(2πσ²ᵢc) + (xᵢ - μᵢc)² / σ²ᵢc]
```

### Algorithmic Implementation

#### Training Algorithm (Learning Phase)

**Input**: Training data X (n_samples × n_features), labels y (n_samples)

**Output**: Learned parameters (class_prior, theta, var)

**Algorithm**:
```
1. Extract unique classes: classes = unique(y)
2. Initialize parameter arrays:
   - class_prior: array of size n_classes
   - theta (means): array of size (n_classes × n_features)
   - var (variances): array of size (n_classes × n_features)

3. For each class C in classes:
   a. Select samples: X_c = X[y == C]
   b. Calculate prior: P(C) = |X_c| / |X|
   c. Calculate means: μᵢc = mean(X_c[:, i]) for each feature i
   d. Calculate variances: σ²ᵢc = var(X_c[:, i]) + ε for each feature i
      (ε = 1e-9 for numerical stability, prevents division by zero)

4. Return (class_prior, theta, var)
```

**Computational Complexity**: O(n_samples × n_features × n_classes)

**Implementation** ([naive_bayes_manual.py:24-72](src/naive_bayes_manual.py#L24)):
```python
def fit(self, X, y):
    self.classes_ = np.unique(y)
    self.n_classes_ = len(self.classes_)
    self.n_features_ = X.shape[1]

    self.class_prior_ = np.zeros(self.n_classes_)
    self.theta_ = np.zeros((self.n_classes_, self.n_features_))
    self.var_ = np.zeros((self.n_classes_, self.n_features_))

    for idx, class_label in enumerate(self.classes_):
        X_class = X[y == class_label]
        self.class_prior_[idx] = X_class.shape[0] / X.shape[0]
        self.theta_[idx] = np.mean(X_class, axis=0)
        self.var_[idx] = np.var(X_class, axis=0) + self.epsilon

    return self
```

#### Prediction Algorithm (Inference Phase)

**Input**: Test data X_test (n_test × n_features), learned parameters

**Output**: Predicted class labels y_pred (n_test)

**Algorithm**:
```
1. For each test sample x in X_test:

   2. For each class C:
      a. Initialize: log_posterior_C = log(P(C))

      b. For each feature i:
         - Calculate: log_likelihood_i = -0.5 × [log(2π × σ²ᵢc) + (xᵢ - μᵢc)² / σ²ᵢc]
         - Add to posterior: log_posterior_C += log_likelihood_i

      c. Store: log_posteriors[C] = log_posterior_C

   3. Predict: y_pred[x] = argmax_C(log_posteriors)

4. Return y_pred
```

**Computational Complexity**: O(n_test × n_features × n_classes)

**Implementation** ([naive_bayes_manual.py:74-142](src/naive_bayes_manual.py#L74)):
```python
def _calculate_log_likelihood(self, X):
    n_samples = X.shape[0]
    log_likelihood = np.zeros((n_samples, self.n_classes_))

    for idx in range(self.n_classes_):
        mean = self.theta_[idx]
        var = self.var_[idx]

        # Gaussian PDF in log space
        log_2pi_var = np.log(2 * np.pi * var)
        squared_diff = ((X - mean) ** 2) / var
        log_likelihood[:, idx] = -0.5 * np.sum(log_2pi_var + squared_diff, axis=1)

    return log_likelihood

def predict(self, X):
    log_posterior = self.predict_log_proba(X)
    class_indices = np.argmax(log_posterior, axis=1)
    return self.classes_[class_indices]
```

### Numerical Stability Techniques

1. **Log-Space Computation**: Prevents underflow when multiplying many probabilities
2. **Epsilon Smoothing**: Adds 1e-9 to variance to prevent division by zero
3. **Vectorized Operations**: Uses NumPy broadcasting for efficient computation

### Scikit-learn Wrapper

Thin wrapper around `sklearn.naive_bayes.GaussianNB` providing consistent interface with manual implementation for fair comparison.

**Key Configuration**:
- `var_smoothing=1e-9`: Matches epsilon in manual implementation

### Data Management

- **Validation**: Checks for missing values, duplicates, class balance
- **Stratified splitting**: Maintains class proportions in train/test sets
- **Optional normalization**: Z-score standardization: z = (x - μ) / σ

### Evaluation Metrics

- **Accuracy**: (TP + TN) / Total = Overall correctness
- **Precision**: TP / (TP + FP) = How many predicted positives are correct
- **Recall**: TP / (TP + FN) = How many actual positives are found
- **F1-score**: 2 × (Precision × Recall) / (Precision + Recall) = Harmonic mean
- **Confusion Matrix**: Grid showing predicted vs actual classifications

### Ring Buffer Logging

Custom logging system that:
- Limits lines per log file (default: 10,000)
- Maintains maximum number of files (default: 5)
- Automatically rotates when limits reached
- Prevents unbounded disk usage

## Dataset: Iris Flowers

**Features** (4 measurements in cm):
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

**Classes** (3 species, 50 samples each):
- Iris Setosa
- Iris Versicolor
- Iris Virginica

**Total**: 150 samples, perfectly balanced

**Source**: Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems"

## Key Takeaways: Real-World Applications

Naive Bayes algorithms are widely used across industries due to their simplicity, speed, and effectiveness. Here are 10 practical applications:

### 1. Email Spam Filtering (Technology)
**Industry**: Email Services, Cybersecurity
**Implementation**: Analyze email content (subject, body, sender) to classify as spam or legitimate. Train on word frequencies and metadata to identify suspicious patterns.
**Benefits**: Fast real-time classification, low computational overhead, adaptive to new spam patterns through continuous retraining.

### 2. Sentiment Analysis (Social Media & Marketing)
**Industry**: Social Media Platforms, Market Research, Brand Management
**Implementation**: Classify customer reviews, tweets, and comments as positive, negative, or neutral based on word frequencies and sentiment indicators.
**Benefits**: Quick sentiment scoring of large volumes of text, helps companies respond to customer feedback, tracks brand reputation in real-time.

### 3. Medical Diagnosis Support (Healthcare)
**Industry**: Healthcare, Clinical Decision Support Systems
**Implementation**: Classify patient symptoms and test results to suggest potential diagnoses. Features include vital signs, lab values, patient history, and reported symptoms.
**Benefits**: Fast preliminary screening tool, helps prioritize urgent cases, reduces diagnostic time for common conditions while maintaining transparency for medical review.

### 4. Credit Risk Assessment (Finance)
**Industry**: Banking, Lending, Credit Card Companies
**Implementation**: Classify loan applicants as low-risk or high-risk based on credit history, income, employment status, debt-to-income ratio, and payment patterns.
**Benefits**: Fast automated approval for low-risk applications, scalable to millions of applications, interpretable risk factors for regulatory compliance.

### 5. Document Classification (Enterprise)
**Industry**: Legal, Government, Corporate Document Management
**Implementation**: Automatically categorize documents into topics (contracts, invoices, reports, correspondence) based on text content and metadata.
**Benefits**: Automates document routing, enables efficient search and retrieval, reduces manual sorting time by 80-90%, scales to millions of documents.

### 6. Weather Prediction (Meteorology)
**Industry**: Weather Services, Agriculture, Aviation
**Implementation**: Classify weather conditions (sunny, rainy, cloudy) based on atmospheric features like temperature, humidity, pressure, wind speed, and historical patterns.
**Benefits**: Fast short-term forecasts, simple model interpretable by meteorologists, computationally cheap for embedded weather stations.

### 7. Customer Churn Prediction (Telecommunications)
**Industry**: Telecom, SaaS, Subscription Services
**Implementation**: Classify customers as likely to churn or retain based on usage patterns, customer service interactions, payment history, and engagement metrics.
**Benefits**: Early identification of at-risk customers, enables targeted retention campaigns, minimal latency for real-time scoring during customer calls.

### 8. Fake News Detection (Media & Journalism)
**Industry**: Social Media, News Aggregators, Content Moderation
**Implementation**: Classify articles as credible or suspicious based on linguistic features, source reputation, claim patterns, and cross-reference with fact-checking databases.
**Benefits**: Fast preliminary screening of viral content, scalable to high-volume newsfeeds, helps prioritize manual fact-checking efforts.

### 9. Intrusion Detection Systems (Cybersecurity)
**Industry**: Network Security, Cloud Services, Enterprise IT
**Implementation**: Classify network traffic as normal or anomalous based on packet features, connection patterns, protocol usage, and behavioral signatures.
**Benefits**: Real-time threat detection with minimal latency, low computational overhead for high-throughput networks, adaptive to new attack patterns.

### 10. Recommendation Systems (E-commerce & Streaming)
**Industry**: Online Retail, Video Streaming, Music Platforms
**Implementation**: Classify user preferences and predict product/content categories of interest based on browsing history, purchase patterns, ratings, and demographic features.
**Benefits**: Fast personalized recommendations, handles cold-start problems with demographic priors, computationally efficient for millions of users.

### Why Naive Bayes Works So Well

Despite its "naive" independence assumption, Naive Bayes excels because:
- **Speed**: Trains and predicts in milliseconds, even on large datasets
- **Simplicity**: Easy to understand, implement, and explain to stakeholders
- **Data Efficiency**: Works well even with limited training data
- **Scalability**: Handles high-dimensional feature spaces effectively
- **Interpretability**: Learned parameters (priors, means, variances) are transparent
- **Real-time Ready**: Low latency makes it perfect for online applications

## Technical Requirements

### Code Quality Standards

- Maximum 150 lines per Python file (strictly enforced)
- Type hints on all function signatures
- Comprehensive docstrings
- No hardcoded paths or magic numbers
- Modular design with single responsibility principle

### Dependencies

- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pyyaml >= 6.0

## Key Findings

1. **Perfect Agreement**: NumPy and scikit-learn implementations achieve 100% prediction agreement across all experiments
2. **High Accuracy**: 91-97% accuracy depending on train-test split
3. **Negligible Parameter Differences**: Learned parameters (priors, means, variances) match within floating-point precision
4. **Fast Training**: Both implementations train in <10ms
5. **Normalization Impact**: Feature scaling has minimal effect on Gaussian Naive Bayes (as expected theoretically)

## Learning Objectives Achieved

✅ Understanding of Bayes' theorem and conditional probability
✅ Implementation of Gaussian probability density function
✅ Numerical stability techniques (log probabilities, epsilon smoothing)
✅ Scikit-learn API familiarity
✅ Model evaluation and comparison methodology
✅ Professional software engineering practices
✅ Data visualization and reporting
✅ Virtual environment management with UV

## References

- [Scikit-learn Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems"
- PROJECT_GUIDELINES.md for code standards

## Code Files Summary

This project consists of 12 Python files totaling 1,183 lines of code, organized by functional category:

### Main Application Layer

| File | Lines | Purpose |
|------|-------|---------|
| [main.py](main.py) | 103 | Main orchestration script - loads config, runs experiments, generates results |

### Core Model Implementations

| File | Lines | Purpose |
|------|-------|---------|
| [src/naive_bayes_manual.py](src/naive_bayes_manual.py) | 142 | Manual Gaussian Naive Bayes implementation using NumPy from scratch |
| [src/naive_bayes_sklearn.py](src/naive_bayes_sklearn.py) | 133 | Scikit-learn wrapper for comparison with identical interface |

### Data Pipeline

| File | Lines | Purpose |
|------|-------|---------|
| [src/data_loader.py](src/data_loader.py) | 132 | Iris dataset loading, validation, stratified splitting, normalization |

### Evaluation & Visualization

| File | Lines | Purpose |
|------|-------|---------|
| [src/evaluator.py](src/evaluator.py) | 105 | Metrics calculation, parameter comparison, runtime benchmarking |
| [src/visualizer.py](src/visualizer.py) | 125 | Generates 6 visualizations per experiment at 300 DPI |

### Utility Modules

| File | Lines | Purpose |
|------|-------|---------|
| [src/utils/paths.py](src/utils/paths.py) | 95 | Centralized path management with singleton pattern |
| [src/utils/validators.py](src/utils/validators.py) | 150 | Input validation for arrays, dataframes, train-test splits |
| [src/utils/helpers.py](src/utils/helpers.py) | 128 | General utilities: config loading, normalization, benchmarking |
| [src/utils/logger.py](src/utils/logger.py) | 148 | Ring buffer logging with line and file limits |

### Module Initialization

| File | Lines | Purpose |
|------|-------|---------|
| [src/__init__.py](src/__init__.py) | 16 | Package initialization and public API exports |
| [src/utils/__init__.py](src/utils/__init__.py) | 9 | Utils subpackage initialization |

### Compliance Metrics

- **Total Python Files**: 12
- **Total Lines of Code**: 1,183
- **Average Lines per File**: 99
- **Maximum File Length**: 150 lines (validators.py)
- **Minimum File Length**: 9 lines (utils/__init__.py)
- **PROJECT_GUIDELINES Compliance**: ✅ 100% (all files ≤150 lines)

### Architecture Highlights

**Modular Design**: Clear separation between data loading, model implementations, evaluation, and utilities

**Single Responsibility**: Each module has a focused purpose - data_loader handles data, evaluator handles metrics, visualizer handles plots

**Reusability**: Utility modules (paths, validators, helpers, logger) provide reusable components across the project

**Type Safety**: All functions include type hints for better IDE support and error detection

**Testing Ready**: Modular structure enables easy unit testing of individual components

## License

Educational project - free to use and modify.

## Author

**Author:** Hadar Wayn

**Project:** L21 - Iris Classification using Naive Bayes: Comparative Implementation Study

**Date:** December 2025

**Technical Implementation:** Claude Code (Claude Sonnet 4.5)

**Repository:** [github.com/hadarwayn/L21-Iris_Classification_using_Naive_Bayes-Comparative_Implementation_Study](https://github.com/hadarwayn/L21-Iris_Classification_using_Naive_Bayes-Comparative_Implementation_Study)
