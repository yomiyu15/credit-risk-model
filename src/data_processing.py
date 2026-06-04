"""
Feature engineering and proxy target construction for credit risk modeling.

Uses RFM (Recency, Frequency, Monetary) segmentation to define a Basel II–aligned
proxy for default when no historical default labels exist in transaction data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Columns present in Xente transaction data
TRANSACTION_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
    "FraudResult",
]

# Customer-level features used for modeling (interpretable aggregates)
MODEL_FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_transaction_value",
    "transaction_value_std",
    "max_transaction_value",
    "channel_diversity",
    "product_category_diversity",
    "fraud_rate",
    "pay_later_share",
    "debit_share",
    "pricing_strategy_mean",
    "tenure_days",
    "transactions_per_day",
]


# ============================================================================
# Custom Transformers for sklearn Pipeline
# ============================================================================

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features (hour, day, month, year) from timestamp columns."""

    def __init__(self, timestamp_col: str = "TransactionStartTime"):
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.timestamp_col not in X.columns:
            logger.warning(
                f"Column {self.timestamp_col} not found; skipping temporal extraction")
            return X

        X[self.timestamp_col] = pd.to_datetime(
            X[self.timestamp_col], errors="coerce")
        X["TransactionHour"] = X[self.timestamp_col].dt.hour
        X["TransactionDay"] = X[self.timestamp_col].dt.day
        X["TransactionMonth"] = X[self.timestamp_col].dt.month
        X["TransactionYear"] = X[self.timestamp_col].dt.year
        X["TransactionDayofWeek"] = X[self.timestamp_col].dt.dayofweek

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-Hot encode categorical columns; keep original columns."""

    def __init__(self, categorical_cols: list[str] | None = None, drop: str = "first"):
        self.categorical_cols = categorical_cols or []
        self.drop = drop
        self.encoder_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if not self.categorical_cols:
            # Auto-detect categorical columns if not specified
            self.categorical_cols = X.select_dtypes(
                include=["object"]).columns.tolist()

        # Filter to only existing columns
        self.categorical_cols = [
            col for col in self.categorical_cols if col in X.columns]

        if self.categorical_cols:
            self.encoder_ = OneHotEncoder(
                drop=self.drop,
                sparse_output=False,
                handle_unknown="ignore",
            )
            self.encoder_.fit(X[self.categorical_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not self.categorical_cols or self.encoder_ is None:
            return X

        encoded = self.encoder_.transform(X[self.categorical_cols])
        feature_names = self.encoder_.get_feature_names_out(
            self.categorical_cols)

        encoded_df = pd.DataFrame(
            encoded, columns=feature_names, index=X.index)
        X = pd.concat([X, encoded_df], axis=1)
        return X


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values via imputation or removal."""

    def __init__(self, strategy: str = "median", threshold: float = 0.5):
        """
        Args:
            strategy: 'mean', 'median', 'most_frequent', or 'drop'
            threshold: drop columns with >threshold missing fraction
        """
        self.strategy = strategy
        self.threshold = threshold
        self.imputer_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # Drop columns with too many missing values
        missing_frac = X.isnull().sum() / len(X)
        self.cols_to_drop_ = missing_frac[missing_frac >
                                          self.threshold].index.tolist()

        X = X.drop(columns=self.cols_to_drop_, errors="ignore")

        # Fit imputer on numerical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self.imputer_ = SimpleImputer(strategy=self.strategy)
            self.imputer_.fit(X[numeric_cols])

        self.numeric_cols_ = numeric_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(columns=self.cols_to_drop_, errors="ignore")

        if self.imputer_ is not None and self.numeric_cols_:
            X[self.numeric_cols_] = self.imputer_.transform(
                X[self.numeric_cols_])

        return X


class NumericalScaler(BaseEstimator, TransformerMixin):
    """Standardize numerical features to mean=0, std=1."""

    def __init__(self):
        self.scaler_ = None
        self.numeric_cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols_ = numeric_cols

        if numeric_cols:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[numeric_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.scaler_ is not None and self.numeric_cols_:
            X[self.numeric_cols_] = self.scaler_.transform(
                X[self.numeric_cols_])
        return X


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    High-level feature engineering pipeline combining temporal, categorical,
    and frequency-based features. Compatible with sklearn fit/transform API.

    This is the main interface for transaction-level feature engineering.
    """

    def __init__(self, standardize: bool = False):
        self.standardize = standardize
        self.pipeline_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit all transformers in the pipeline."""
        X = X.copy()

        # Detect categorical columns automatically
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        # Remove timestamp columns from categorical
        categorical_cols = [
            c for c in categorical_cols if 'time' not in c.lower()]

        steps = [
            ("temporal", TemporalFeatureExtractor(
                timestamp_col="TransactionStartTime")),
            ("missing_values", MissingValueHandler(strategy="median")),
            ("categorical", CategoricalEncoder(categorical_cols=categorical_cols)),
        ]

        if self.standardize:
            steps.append(("scaling", NumericalScaler()))

        self.pipeline_ = Pipeline(steps, verbose=False)
        self.pipeline_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply pipeline transformations and compute frequency features."""
        if self.pipeline_ is None:
            raise RuntimeError("Pipeline must be fitted before transform")

        X_orig = X.copy()

        # Apply pipeline transformations
        X = self.pipeline_.transform(X_orig)

        # Add frequency features for categorical columns
        freq_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
        for col in freq_cols:
            if col in X_orig.columns:
                freq_name = f"{col}_freq"
                X[freq_name] = X_orig[col].map(
                    X_orig[col].value_counts()).fillna(1)

        # Add ratio feature if both Amount and Value exist
        if 'Amount' in X.columns and 'Value' in X.columns:
            X['Amount_Value_Ratio'] = (
                X['Amount'] / (X['Value'] + 1e-6)
            )

        # Preserve target variable if present
        if 'is_high_risk' in X_orig.columns:
            X['is_high_risk'] = X_orig['is_high_risk']

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def build_preprocessing_pipeline(
    categorical_cols: list[str] | None = None,
    imputation_strategy: str = "median",
    standardize: bool = True,
) -> Pipeline:
    """
    Build a complete preprocessing pipeline for transaction-level data.

    Args:
        categorical_cols: List of categorical column names to encode.
        imputation_strategy: Strategy for missing value imputation.
        standardize: Whether to standardize numerical features.

    Returns:
        sklearn Pipeline that transforms raw transaction data.
    """
    steps = [
        ("temporal", TemporalFeatureExtractor(
            timestamp_col="TransactionStartTime")),
        ("missing_values", MissingValueHandler(strategy=imputation_strategy)),
        ("categorical", CategoricalEncoder(categorical_cols=categorical_cols)),
    ]

    if standardize:
        steps.append(("scaling", NumericalScaler()))

    pipeline = Pipeline(steps, verbose=False)
    logger.info("Built preprocessing pipeline with %d steps", len(steps))
    return pipeline


# ============================================================================
# Data Loading and RFM-based Target Construction
# ============================================================================


def load_raw_transactions(path: str | Path) -> pd.DataFrame:
    """Load raw transaction CSV (training.csv or equivalent)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Download Xente data from the challenge portal and place it in data/raw/."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    logger.info("Loaded %d transactions from %s", len(df), path)
    return df


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce"
    )
    return df.dropna(subset=["TransactionStartTime"])


def compute_rfm(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary metrics per customer.

    Recency: days since last transaction (higher = less engaged).
    Frequency: transaction count.
    Monetary: sum of absolute transaction values.
    """
    df = _parse_timestamps(df)
    if reference_date is None:
        reference_date = df["TransactionStartTime"].max()

    customer_id = "CustomerId"
    grouped = df.groupby(customer_id)

    rfm = pd.DataFrame(
        {
            "recency_days": (
                reference_date - grouped["TransactionStartTime"].max()
            ).dt.days,
            "frequency": grouped.size(),
            "monetary_total": grouped["Value"].sum(),
        }
    )
    return rfm.reset_index()


def engineer_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data to customer-level model features."""
    df = _parse_timestamps(df)
    reference_date = df["TransactionStartTime"].max()
    min_date = df["TransactionStartTime"].min()
    customer_id = "CustomerId"

    rfm = compute_rfm(df, reference_date)
    grouped = df.groupby(customer_id)

    features = rfm.set_index(customer_id)
    features["avg_transaction_value"] = grouped["Value"].mean()
    features["transaction_value_std"] = grouped["Value"].std().fillna(0)
    features["max_transaction_value"] = grouped["Value"].max()
    features["channel_diversity"] = grouped["ChannelId"].nunique()
    features["product_category_diversity"] = grouped["ProductCategory"].nunique()

    if "FraudResult" in df.columns:
        features["fraud_rate"] = grouped["FraudResult"].mean()
    else:
        features["fraud_rate"] = 0.0

    channel_lower = df["ChannelId"].astype(str).str.lower()
    pay_later_mask = channel_lower.str.contains("pay", na=False) | channel_lower.str.contains(
        "later", na=False
    )
    pay_later = (
        df.assign(_pay_later=pay_later_mask)
        .groupby(customer_id)["_pay_later"]
        .mean()
    )
    features["pay_later_share"] = pay_later

    debit_mask = df["Amount"] > 0
    features["debit_share"] = (
        df.assign(_debit=debit_mask).groupby(customer_id)["_debit"].mean()
    )

    if "PricingStrategy" in df.columns:
        features["pricing_strategy_mean"] = grouped["PricingStrategy"].mean()
    else:
        features["pricing_strategy_mean"] = 0.0

    first_tx = grouped["TransactionStartTime"].min()
    features["tenure_days"] = (reference_date - first_tx).dt.days.clip(lower=1)
    features["transactions_per_day"] = features["frequency"] / \
        features["tenure_days"]

    return features.reset_index()


def create_proxy_target(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Label high-risk customers via RFM clustering.

    Business rule: cluster with worst combined RFM profile (high recency,
    low frequency, low monetary) is assigned is_high_risk=1 (bad).
    This proxy aligns with churn/inactivity risk as a stand-in for credit default
    when no loan performance history exists.
    """
    rfm = rfm.copy()
    rfm_cols = ["recency_days", "frequency", "monetary_total"]
    X = rfm[rfm_cols].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init=10)
    rfm["rfm_cluster"] = kmeans.fit_predict(X_scaled)

    # Risk score: higher recency and lower frequency/monetary => higher risk
    cluster_profiles = rfm.groupby("rfm_cluster")[rfm_cols].mean()
    cluster_profiles["risk_score"] = (
        cluster_profiles["recency_days"]
        - cluster_profiles["frequency"] /
        (cluster_profiles["frequency"].max() + 1)
        - cluster_profiles["monetary_total"]
        / (cluster_profiles["monetary_total"].max() + 1)
    )
    bad_cluster = int(cluster_profiles["risk_score"].idxmax())
    rfm["is_high_risk"] = (rfm["rfm_cluster"] == bad_cluster).astype(int)

    logger.info(
        "Proxy target: cluster %d labeled high-risk (%.1f%% of customers)",
        bad_cluster,
        100 * rfm["is_high_risk"].mean(),
    )
    return rfm


def compute_weight_of_evidence(
    df: pd.DataFrame,
    feature: str,
    target: str = "is_high_risk",
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """
    Compute WoE and Information Value for a numeric feature.

    IV interpretation: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, >0.3 strong.
    """
    work = df[[feature, target]].dropna().copy()
    work["bin"] = pd.qcut(work[feature], q=min(
        n_bins, work[feature].nunique()), duplicates="drop")

    grouped = work.groupby("bin", observed=True)[target].agg(["sum", "count"])
    grouped.columns = ["bad", "total"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()
    eps = 1e-6

    grouped["dist_bad"] = (grouped["bad"] + eps) / (total_bad + eps)
    grouped["dist_good"] = (grouped["good"] + eps) / (total_good + eps)
    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
    grouped["iv_component"] = (
        grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

    iv = float(grouped["iv_component"].sum())
    return grouped.reset_index(), iv


def select_features_by_iv(
    df: pd.DataFrame,
    features: list[str],
    target: str = "is_high_risk",
    min_iv: float = 0.02,
) -> tuple[list[str], pd.DataFrame]:
    """Return features with IV >= min_iv and full IV summary table."""
    rows = []
    selected = []
    for feat in features:
        if feat not in df.columns or df[feat].nunique() < 2:
            continue
        try:
            _, iv = compute_weight_of_evidence(df, feat, target=target)
            rows.append({"feature": feat, "iv": iv})
            if iv >= min_iv:
                selected.append(feat)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping IV for %s: %s", feat, exc)

    iv_table = pd.DataFrame(rows).sort_values("iv", ascending=False)
    if not selected:
        selected = [f for f in features if f in df.columns]
    return selected, iv_table


def build_modeling_dataset(
    df: pd.DataFrame,
    n_clusters: int = 3,
    min_iv: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full pipeline: RFM -> proxy target -> customer features -> IV selection.
    """
    rfm = compute_rfm(df)
    rfm_labeled = create_proxy_target(rfm, n_clusters=n_clusters)
    features = engineer_customer_features(df)

    dataset = features.merge(
        rfm_labeled[["CustomerId", "is_high_risk", "rfm_cluster"]],
        on="CustomerId",
    )

    selected, iv_table = select_features_by_iv(
        dataset, MODEL_FEATURE_COLUMNS, min_iv=min_iv
    )

    metadata = {
        "n_customers": len(dataset),
        "default_rate": float(dataset["is_high_risk"].mean()),
        "selected_features": selected,
        "iv_summary": iv_table.to_dict(orient="records"),
        "n_clusters": n_clusters,
    }
    return dataset, metadata


def save_processed_data(
    dataset: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Persist customer-level modeling table and metadata JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_dir / "customer_features.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved processed data to %s", output_dir)


def run_processing_pipeline(
    raw_path: str | Path,
    output_dir: str | Path,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """CLI entrypoint for feature engineering."""
    df = load_raw_transactions(raw_path)
    dataset, metadata = build_modeling_dataset(df, n_clusters=n_clusters)
    save_processed_data(dataset, metadata, output_dir)
    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="Process Xente transactions")
    parser.add_argument(
        "--input",
        default="data/raw/data.csv",
        help="Path to raw transaction CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for processed features",
    )
    parser.add_argument("--clusters", type=int, default=3)
    args = parser.parse_args()
    run_processing_pipeline(args.input, args.output, n_clusters=args.clusters)
