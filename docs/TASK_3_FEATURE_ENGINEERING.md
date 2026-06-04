# Task 3: Feature Engineering – Implementation Guide

## Overview
Task 3 implements a **production-ready, reproducible feature engineering pipeline** that transforms raw transaction-level data into a model-ready dataset. The implementation uses **sklearn Pipeline** to chain together all transformation steps, ensuring reproducibility and scalability.

## Architecture

### Core Components

#### 1. **Custom Transformers** (`src/data_processing.py`)
Each transformer follows sklearn's `BaseEstimator` and `TransformerMixin` interface:

- **TemporalFeatureExtractor**
  - Extracts temporal features from `TransactionStartTime` column
  - Features: `transaction_hour`, `transaction_day`, `transaction_month`, `transaction_year`, `transaction_dayofweek`
  - Handles invalid timestamps gracefully

- **CategoricalEncoder**
  - One-Hot encodes categorical variables
  - Auto-detects categorical columns if not specified
  - Handles unknown categories during transform (critical for production)

- **MissingValueHandler**
  - Drops columns with >50% missing values (configurable threshold)
  - Imputes remaining missing values using specified strategy (mean, median, most_frequent)
  - Only operates on numerical columns

- **NumericalScaler**
  - Standardizes numerical features to mean=0, std=1
  - Uses sklearn's `StandardScaler` internally
  - Necessary for distance-based algorithms and fair feature importance

#### 2. **build_preprocessing_pipeline()**
Chains transformers into a single sklearn Pipeline:
```python
pipe = build_preprocessing_pipeline(
    categorical_cols=["ChannelId", "ProductCategory"],
    imputation_strategy="median",
    standardize=True
)
processed = pipe.fit_transform(raw_data)
```

**Steps in order:**
1. Temporal feature extraction
2. Missing value handling
3. Categorical encoding
4. Numerical standardization (optional)

#### 3. **Customer-Level Aggregation**
In addition to transaction-level transformations:

- **engineer_customer_features()**: Aggregates to customer level
  - Recency, Frequency, Monetary (RFM)
  - Transaction statistics (average, std, max)
  - Channel and category diversity
  - Fraud rate, payment channel preferences
  - Tenure and transaction frequency per day

- **compute_rfm()**: Computes RFM metrics
  - Recency = days since last transaction
  - Frequency = transaction count
  - Monetary = sum of transaction values

#### 4. **Proxy Target Variable**
- **create_proxy_target()**: RFM-based clustering
  - Clusters customers into N groups (default: 4)
  - Assigns highest-risk cluster as `default_risk=1` (least engaged)
  - Business justification: churn/inactivity ≈ default risk (Basel II proxy)

#### 5. **Feature Selection**
- **select_features_by_iv()**: Information Value filtering
  - Computes WoE (Weight of Evidence) for each feature
  - Calculates IV (Information Value)
  - Selects features with IV >= min_iv (default: 0.02)
  - **IV Interpretation:**
    - < 0.02: Useless
    - 0.02 - 0.1: Weak
    - 0.1 - 0.3: Medium
    - > 0.3: Strong

## Deliverables

### Files Modified/Created

| File | Purpose |
|------|---------|
| `src/data_processing.py` | Core pipeline and transformers |
| `tests/test_feature_engineering.py` | Transformer unit tests (19 tests) |
| `tests/test_data_processing.py` | Data processing integration tests |
| `scripts/demo_feature_engineering.py` | End-to-end demonstration |

### Key Features

✅ **Reproducibility**
- All stochastic operations use `random_state=42`
- Pipeline is serializable with joblib

✅ **Production-Ready**
- Handles unknown categories with `handle_unknown="ignore"`
- Robust missing value handling
- Proper sklearn API compliance

✅ **Interpretability**
- Feature names preserved through pipeline
- WoE-based feature selection aligns with Basel II
- Detailed logging at each step

✅ **Scalability**
- Works with small samples or full datasets
- Memory-efficient transform operations
- Modular design enables custom transformations

## Usage Examples

### Basic Pipeline Usage
```python
from src.data_processing import build_preprocessing_pipeline

# Create and fit pipeline
pipe = build_preprocessing_pipeline(
    categorical_cols=["ChannelId", "ProductCategory"],
    standardize=True
)

# Transform data
processed = pipe.fit_transform(raw_transactions)
```

### Full Workflow
```python
from src.data_processing import (
    load_raw_transactions,
    engineer_customer_features,
    compute_rfm,
    create_proxy_target,
    select_features_by_iv,
    build_modeling_dataset
)

# Load and process
transactions = load_raw_transactions("data/raw/data.csv")
dataset, metadata = build_modeling_dataset(transactions, n_clusters=4)

# dataset now contains:
# - Customer ID
# - All engineered features
# - Target: default_risk (0 or 1)
# metadata contains:
# - Selected features for modeling
# - Information Value rankings
# - Clustering details
```

### CLI Usage
```bash
# Process data from command line
python src/data_processing.py \
    --input data/raw/data.csv \
    --output data/processed \
    --clusters 4
```

## Testing

### Run All Tests
```bash
# Feature engineering tests
pytest tests/test_feature_engineering.py -v

# Data processing tests
pytest tests/test_data_processing.py -v

# All tests
pytest tests/ -v
```

### Test Coverage
- **19 tests** in `test_feature_engineering.py`
- **6 tests** in `test_data_processing.py`
- Coverage includes:
  - Temporal extraction (hour, day, month, year ranges)
  - Categorical encoding (One-Hot behavior)
  - Missing value imputation
  - Standardization verification
  - RFM computation correctness
  - Proxy target binary validation
  - Feature aggregation accuracy

## Design Decisions & Basel II Alignment

### 1. Why sklearn Pipeline?
- **Reproducibility**: Fitted pipeline can be saved and applied consistently
- **No Data Leakage**: Transformers fit only on training data
- **Flexibility**: Easy to swap transformers or add new ones
- **Production Deployment**: Pipeline serializes for REST API use

### 2. Why RFM-Based Proxy?
- **Problem**: Raw data has no default labels (unobserved)
- **Solution**: Use behavioral engagement as proxy
- **Justification**: Inactive/disengaged customers (high recency, low frequency/monetary) are riskier
- **Basel II Alignment**: Focuses on observable customer behavior, not historical defaults

### 3. Why One-Hot Encoding?
- Handles variable-length categorical values
- Produces interpretable binary features
- Works well with logistic regression (interpretable) and tree-based models (scalable)
- `handle_unknown="ignore"` handles production scoring on unseen categories

### 4. Why Standardization?
- Required for distance-based algorithms (KMeans, etc.)
- Fair feature importance in linear models
- Stabilizes neural networks (future extension)
- Backward-compatible: non-scaled versions available

### 5. Why Information Value for Selection?
- **Interpretability**: WoE-based features are explainable under Basel II
- **Statistical rigor**: IV accounts for both separation and entropy
- **Parsimony**: Selects only predictive features, reduces overfitting
- **Regulatory**: Aligns with credit scoring best practices

## Integration with Downstream Tasks

### Task 4: Proxy Target
- Uses `create_proxy_target()` and `compute_rfm()`
- Integrates cluster labels into modeling dataset

### Task 5: Model Training
- Uses selected features from IV ranking
- Preprocessing pipeline can be frozen in production model

### Task 6: API Deployment
- Pipeline is serialized and loaded by FastAPI service
- `/predict` endpoint applies same transformations to new data

## Limitations & Future Work

### Current Limitations
1. **Categorical handling**: Assumes low-cardinality categorical features
2. **Temporal aggregation**: Uses single snapshot date; doesn't handle rolling windows
3. **Outliers**: No explicit outlier removal (can be added as transformer)
4. **Non-linear relationships**: WoE binning assumes monotonic relationship with target

### Possible Extensions
- **Auto-encoding**: Deep learning for feature extraction
- **Polynomial features**: For non-linear relationships
- **Feature interactions**: Cross-product features
- **Domain-specific transformers**: E.g., fraud-specific ratios
- **Feature store**: Database for feature versioning and caching

## References
- sklearn Pipeline: https://scikit-learn.org/stable/modules/compose.html#pipeline
- Weight of Evidence: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
- Basel II Credit Risk: https://fastercapital.com/content/Basel-Accords--What-They-Are-and-How-They-Affect-Credit-Risk-Management.html
