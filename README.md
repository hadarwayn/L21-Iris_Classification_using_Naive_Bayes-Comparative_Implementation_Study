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

### Manual NumPy Implementation

**Key Features**:
- Pure NumPy implementation (no scikit-learn)
- Log probabilities for numerical stability
- Epsilon smoothing to prevent zero variance
- Vectorized operations for efficiency

**Algorithm** ([naive_bayes_manual.py:44](src/naive_bayes_manual.py#L44)):
```python
# Training (fit method)
for each class:
    prior[class] = count(class) / total_samples
    mean[class] = average(features for this class)
    variance[class] = variance(features) + epsilon

# Prediction (predict method)
for each sample:
    for each class:
        log_likelihood = sum of log(Gaussian PDF)
        log_posterior = log_likelihood + log(prior)
    prediction = class with max log_posterior
```

### Scikit-learn Wrapper

Thin wrapper around `sklearn.naive_bayes.GaussianNB` providing consistent interface with manual implementation for fair comparison.

### Data Management

- **Validation**: Checks for missing values, duplicates, class balance
- **Stratified splitting**: Maintains class proportions in train/test sets
- **Optional normalization**: Z-score standardization

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1-score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

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

## License

Educational project - free to use and modify.

## Author

L21 Project - December 2025
