"""Unit tests for feature engineering transformers and pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_processing import (
    TemporalFeatureExtractor,
    CategoricalEncoder,
    MissingValueHandler,
    NumericalScaler,
    build_preprocessing_pipeline,
    engineer_customer_features,
    compute_rfm,
    create_proxy_target,
)


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n_rows = 100
    return pd.DataFrame({
        "TransactionId": [f"TX_{i}" for i in range(n_rows)],
        "CustomerId": [f"CUST_{i % 10}" for i in range(n_rows)],
        "Amount": np.random.randint(1, 100, n_rows),
        "Value": np.random.randint(10, 1000, n_rows),
        "TransactionStartTime": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
        "ChannelId": np.random.choice(["web", "android", "ios", "checkout"], n_rows),
        "ProductCategory": np.random.choice(["Electronics", "Clothing", "Food"], n_rows),
        "FraudResult": np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
        "PricingStrategy": np.random.choice([1, 2, 3], n_rows),
    })


class TestTemporalFeatureExtractor:
    """Test temporal feature extraction transformer."""

    def test_extracts_temporal_features(self, sample_transactions):
        """Test that temporal features are correctly extracted."""
        extractor = TemporalFeatureExtractor()
        result = extractor.fit_transform(sample_transactions)

        assert "transaction_hour" in result.columns
        assert "transaction_day" in result.columns
        assert "transaction_month" in result.columns
        assert "transaction_year" in result.columns
        assert "transaction_dayofweek" in result.columns
        assert "TransactionStartTime" not in result.columns

    def test_hour_range(self, sample_transactions):
        """Test that extracted hours are in valid range [0, 23]."""
        extractor = TemporalFeatureExtractor()
        result = extractor.fit_transform(sample_transactions)

        assert result["transaction_hour"].min() >= 0
        assert result["transaction_hour"].max() <= 23

    def test_returns_dataframe(self, sample_transactions):
        """Test that transformer returns a DataFrame."""
        extractor = TemporalFeatureExtractor()
        result = extractor.fit_transform(sample_transactions)
        assert isinstance(result, pd.DataFrame)


class TestCategoricalEncoder:
    """Test categorical encoding transformer."""

    def test_one_hot_encoding(self, sample_transactions):
        """Test that categorical columns are one-hot encoded."""
        encoder = CategoricalEncoder(
            categorical_cols=["ChannelId", "ProductCategory"])
        result = encoder.fit_transform(sample_transactions)

        # Original categorical columns should be dropped
        assert "ChannelId" not in result.columns
        assert "ProductCategory" not in result.columns

        # One-hot encoded columns should exist
        assert any("ChannelId_" in col for col in result.columns)
        assert any("ProductCategory_" in col for col in result.columns)

    def test_auto_detect_categories(self, sample_transactions):
        """Test that categorical columns are auto-detected if not specified."""
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(sample_transactions)
        assert isinstance(result, pd.DataFrame)


class TestMissingValueHandler:
    """Test missing value handling transformer."""

    def test_imputation(self):
        """Test that missing values are imputed."""
        df = pd.DataFrame({
            "A": [1, 2, np.nan, 4, 5],
            "B": [10, np.nan, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        })
        handler = MissingValueHandler(strategy="median")
        result = handler.fit_transform(df)

        # Numeric columns should have no NaNs
        assert result[["A", "B"]].isna().sum().sum() == 0

    def test_returns_dataframe(self, sample_transactions):
        """Test that handler returns a DataFrame."""
        handler = MissingValueHandler()
        result = handler.fit_transform(sample_transactions)
        assert isinstance(result, pd.DataFrame)


class TestNumericalScaler:
    """Test numerical scaling transformer."""

    def test_standardization(self):
        """Test that numerical features are standardized."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
        })
        scaler = NumericalScaler()
        result = scaler.fit_transform(df)

        # Mean should be close to 0 and std close to 1
        np.testing.assert_almost_equal(result["A"].mean(), 0, decimal=10)
        # Use ddof=0 to match StandardScaler behavior
        np.testing.assert_almost_equal(result["A"].std(ddof=0), 1, decimal=10)


class TestBuildPreprocessingPipeline:
    """Test the complete preprocessing pipeline."""

    def test_pipeline_is_pipeline_object(self, sample_transactions):
        """Test that build_preprocessing_pipeline returns a Pipeline."""
        pipe = build_preprocessing_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_fit_transform(self, sample_transactions):
        """Test that pipeline can fit and transform data."""
        pipe = build_preprocessing_pipeline(
            categorical_cols=["ChannelId", "ProductCategory"]
        )
        result = pipe.fit_transform(sample_transactions)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_transactions)
        # Temporal features should be added
        assert "transaction_hour" in result.columns

    def test_pipeline_without_scaling(self, sample_transactions):
        """Test pipeline without scaling."""
        pipe = build_preprocessing_pipeline(standardize=False)
        result = pipe.fit_transform(sample_transactions)
        assert isinstance(result, pd.DataFrame)


class TestRFMComputation:
    """Test RFM metric computation."""

    def test_rfm_columns_exist(self, sample_transactions):
        """Test that RFM metrics are computed with correct columns."""
        rfm = compute_rfm(sample_transactions)

        assert "CustomerId" in rfm.columns
        assert "recency_days" in rfm.columns
        assert "frequency" in rfm.columns
        assert "monetary_total" in rfm.columns

    def test_rfm_values_are_positive(self, sample_transactions):
        """Test that RFM values are non-negative."""
        rfm = compute_rfm(sample_transactions)

        assert (rfm["recency_days"] >= 0).all()
        assert (rfm["frequency"] > 0).all()
        assert (rfm["monetary_total"] >= 0).all()


class TestCreateProxyTarget:
    """Test proxy target creation via RFM clustering."""

    def test_proxy_target_column_exists(self, sample_transactions):
        """Test that default_risk column is created."""
        rfm = compute_rfm(sample_transactions)
        labeled = create_proxy_target(rfm)

        assert "default_risk" in labeled.columns
        assert "rfm_cluster" in labeled.columns

    def test_binary_target(self, sample_transactions):
        """Test that default_risk is binary (0 or 1)."""
        rfm = compute_rfm(sample_transactions)
        labeled = create_proxy_target(rfm)

        assert set(labeled["default_risk"].unique()).issubset({0, 1})

    def test_clusters_created(self, sample_transactions):
        """Test that clusters are created and assigned."""
        rfm = compute_rfm(sample_transactions)
        labeled = create_proxy_target(rfm, n_clusters=3)

        assert labeled["rfm_cluster"].nunique() <= 3


class TestEngineerCustomerFeatures:
    """Test customer-level feature engineering."""

    def test_customer_features_created(self, sample_transactions):
        """Test that customer-level features are engineered."""
        features = engineer_customer_features(sample_transactions)

        assert "CustomerId" in features.columns
        assert "recency_days" in features.columns
        assert "frequency" in features.columns
        assert "avg_transaction_value" in features.columns
        assert "channel_diversity" in features.columns
        assert "fraud_rate" in features.columns

    def test_one_row_per_customer(self, sample_transactions):
        """Test that features are aggregated to customer level."""
        features = engineer_customer_features(sample_transactions)
        n_unique_customers = sample_transactions["CustomerId"].nunique()

        assert len(features) == n_unique_customers

    def test_features_are_numeric(self, sample_transactions):
        """Test that engineered features are numeric."""
        features = engineer_customer_features(sample_transactions)
        numeric_features = features.select_dtypes(include=[np.number])

        # Most columns should be numeric
        assert len(numeric_features.columns) > 5
