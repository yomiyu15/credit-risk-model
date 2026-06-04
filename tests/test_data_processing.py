import pytest
import pandas as pd
import numpy as np
from src.data_processing import FeaturePipeline


@pytest.fixture
def dummy_transaction_logs():
    """Generates an artificial mini transaction matrix for validation."""
    return pd.DataFrame({
        'TransactionId': [f'T_{i}' for i in range(5)],
        'CustomerId': ['C_1', 'C_1', 'C_2', 'C_2', 'C_1'],
        'TransactionStartTime': [
            '2026-05-01T10:00:00Z',
            '2026-05-01T11:30:00Z',
            '2026-05-02T14:15:00Z',
            '2026-05-03T09:00:00Z',
            '2026-05-04T16:45:00Z'
        ],
        'Amount': [5000.0, -100.0, 25000.0, 1200.0, -500.0],
        'Value': [5000.0, 100.0, 25000.0, 1200.0, 500.0],
        'ProviderId': ['P_1', 'P_2', 'P_1', 'P_1', 'P_2'],
        'ProductId': ['Prod_A', 'Prod_B', 'Prod_A', 'Prod_C', 'Prod_B'],
        'ProductCategory': ['airtime', 'financial_services', 'airtime', 'utility', 'financial_services'],
        'ChannelId': ['Ch_1', 'Ch_2', 'Ch_1', 'Ch_1', 'Ch_2'],
        'is_high_risk': [0, 0, 1, 1, 0]
    })


def test_feature_pipeline_output_integrity(dummy_transaction_logs):
    """Verifies that the FeaturePipeline extracts all core production columns correctly."""
    pipeline = FeaturePipeline()
    processed_df = pipeline.fit_transform(dummy_transaction_logs)

    expected_columns = [
        'Amount', 'Value', 'TransactionHour', 'TransactionDay',
        'TransactionMonth', 'TransactionYear', 'Amount_Value_Ratio',
        'ProviderId_freq', 'ProductId_freq', 'ProductCategory_freq', 'ChannelId_freq', 'is_high_risk'
    ]

    for col in expected_columns:
        assert col in processed_df.columns, f"Required column {col} was dropped or omitted during transforming."


def test_feature_pipeline_no_nan_generation(dummy_transaction_logs):
    """Ensures the data processing transformations do not generate unexpected missing values."""
    pipeline = FeaturePipeline()
    processed_df = pipeline.fit_transform(dummy_transaction_logs)

    assert processed_df.isnull().sum().sum(
    ) == 0, "Pipeline transformation output contains null entries."
