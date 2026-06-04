#!/usr/bin/env python
"""
Demonstration of the complete feature engineering pipeline for Task 3.

This script shows:
1. Loading raw transaction data
2. Building and applying the preprocessing pipeline
3. Computing customer-level RFM features
4. Creating a proxy target variable via RFM clustering
5. Feature selection using Information Value (IV)
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data_processing import (
    load_raw_transactions,
    build_preprocessing_pipeline,
    engineer_customer_features,
    compute_rfm,
    create_proxy_target,
    select_features_by_iv,
    save_processed_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def demonstrate_pipeline():
    """Run the complete feature engineering pipeline."""

    # ========================================================================
    # Step 1: Load raw transaction data
    # ========================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: LOAD RAW TRANSACTION DATA")
    logger.info("=" * 70)

    raw_path = Path("data/raw/data.csv")
    transactions = load_raw_transactions(raw_path)
    logger.info(f"Loaded {len(transactions):,} transactions")
    logger.info(f"Columns: {list(transactions.columns)}")
    logger.info(f"Data types:\n{transactions.dtypes}")

    # ========================================================================
    # Step 2: Build and apply preprocessing pipeline
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: BUILD AND APPLY PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    categorical_cols = ["ChannelId", "ProductCategory", "CurrencyCode"]
    pipeline = build_preprocessing_pipeline(
        categorical_cols=categorical_cols,
        imputation_strategy="median",
        standardize=True,
    )

    # Apply pipeline to a sample for demonstration
    sample_txns = transactions.head(1000).copy()
    processed_sample = pipeline.fit_transform(sample_txns)
    logger.info(f"Processed {len(processed_sample):,} transactions")
    logger.info(f"New columns added: {set(processed_sample.columns) - set(transactions.columns)}")
    logger.info(f"Columns dropped: {set(transactions.columns) - set(processed_sample.columns)}")
    logger.info(f"Processed sample shape: {processed_sample.shape}")
    logger.info(f"\nProcessed sample head:\n{processed_sample.head()}")

    # ========================================================================
    # Step 3: Compute customer-level RFM features
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: COMPUTE CUSTOMER-LEVEL RFM FEATURES")
    logger.info("=" * 70)

    features = engineer_customer_features(transactions)
    logger.info(f"Engineered {len(features):,} unique customers")
    logger.info(f"Feature columns:\n{list(features.columns)}")
    logger.info(f"\nRFM & aggregates summary:\n{features.describe()}")

    # ========================================================================
    # Step 4: Create proxy target variable via RFM clustering
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: CREATE PROXY TARGET VIA RFM CLUSTERING")
    logger.info("=" * 70)

    rfm = compute_rfm(transactions)
    logger.info(f"Computed RFM for {len(rfm):,} customers")
    logger.info(f"\nRFM summary:\n{rfm[['recency_days', 'frequency', 'monetary_total']].describe()}")

    rfm_labeled = create_proxy_target(rfm, n_clusters=4)
    default_rate = rfm_labeled["default_risk"].mean()
    logger.info(f"Proxy default rate: {100 * default_rate:.2f}%")
    logger.info(f"\nCluster distribution:\n{rfm_labeled['rfm_cluster'].value_counts()}")
    logger.info(f"\nDefault risk distribution:\n{rfm_labeled['default_risk'].value_counts()}")

    # ========================================================================
    # Step 5: Feature selection via Information Value
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: FEATURE SELECTION VIA INFORMATION VALUE (IV)")
    logger.info("=" * 70)

    # Merge features with target
    dataset = features.merge(
        rfm_labeled[["CustomerId", "default_risk"]],
        on="CustomerId",
    )

    from src.data_processing import MODEL_FEATURE_COLUMNS
    selected_features, iv_table = select_features_by_iv(
        dataset, MODEL_FEATURE_COLUMNS, min_iv=0.02
    )

    logger.info(f"\nInformation Value (IV) Summary:")
    logger.info(f"Total features evaluated: {len(MODEL_FEATURE_COLUMNS)}")
    logger.info(f"Features selected (IV >= 0.02): {len(selected_features)}")
    logger.info(f"\nIV Ranking:\n{iv_table.to_string(index=False)}")

    # ========================================================================
    # Step 6: Save processed dataset
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: SAVE PROCESSED DATASET")
    logger.info("=" * 70)

    metadata = {
        "n_customers": len(dataset),
        "default_rate": float(dataset["default_risk"].mean()),
        "selected_features": selected_features,
        "iv_summary": iv_table.to_dict(orient="records"),
        "n_clusters": 4,
        "preprocessing_pipeline": {
            "categorical_cols": categorical_cols,
            "imputation_strategy": "median",
            "standardized": True,
        },
    }

    output_dir = Path("data/processed")
    save_processed_data(dataset, metadata, output_dir)
    logger.info(f"Saved to {output_dir}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ENGINEERING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nDataset Summary:")
    logger.info(f"  • Total rows (customers): {len(dataset):,}")
    logger.info(f"  • Total columns: {len(dataset.columns)}")
    logger.info(f"  • Target variable (is_high_risk): Binary (0/1)")
    logger.info(f"  • Default rate: {100 * dataset['default_risk'].mean():.2f}%")
    logger.info(f"  • Selected features for modeling: {len(selected_features)}")
    logger.info(f"\nKey Transformations Applied:")
    logger.info(f"  ✓ Temporal feature extraction (hour, day, month, year)")
    logger.info(f"  ✓ Categorical encoding (One-Hot)")
    logger.info(f"  ✓ Missing value imputation (median)")
    logger.info(f"  ✓ Numerical standardization (mean=0, std=1)")
    logger.info(f"  ✓ RFM-based proxy target creation")
    logger.info(f"  ✓ Feature selection via Information Value (IV)")
    logger.info("\nOutputs:")
    logger.info(f"  • data/processed/customer_features.csv")
    logger.info(f"  • data/processed/metadata.json")

    return dataset, metadata, pipeline


if __name__ == "__main__":
    dataset, metadata, pipeline = demonstrate_pipeline()
