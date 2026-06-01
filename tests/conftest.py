"""Shared fixtures: synthetic Xente-like transaction data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Minimal transaction dataset for unit tests."""
    rng = np.random.default_rng(42)
    n = 500
    customers = [f"C{i % 25}" for i in range(n)]
    base = pd.Timestamp("2019-01-01")
    times = [base + pd.Timedelta(days=int(rng.integers(0, 60))) for _ in range(n)]

    return pd.DataFrame(
        {
            "TransactionId": [f"T{i}" for i in range(n)],
            "BatchId": ["B1"] * n,
            "AccountId": customers,
            "SubscriptionId": customers,
            "CustomerId": customers,
            "CurrencyCode": ["UGX"] * n,
            "CountryCode": [256] * n,
            "ProviderId": ["P1"] * n,
            "ProductId": ["PR1"] * n,
            "ProductCategory": rng.choice(
                ["airtime", "retail", "utility"], n
            ),
            "ChannelId": rng.choice(
                ["web", "android", "paylater"], n
            ),
            "Amount": rng.integers(100, 50000, n).astype(float),
            "Value": rng.integers(100, 50000, n),
            "TransactionStartTime": times,
            "PricingStrategy": rng.integers(0, 3, n),
            "FraudResult": rng.integers(0, 2, n),
        }
    )
