"""Unit tests for feature engineering and proxy target."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import (
    build_modeling_dataset,
    compute_rfm,
    create_proxy_target,
    engineer_customer_features,
    save_processed_data,
    select_features_by_iv,
)


def test_compute_rfm_shape(sample_transactions):
    rfm = compute_rfm(sample_transactions)
    assert len(rfm) == sample_transactions["CustomerId"].nunique()
    assert {"recency_days", "frequency", "monetary_total"}.issubset(rfm.columns)


def test_engineer_customer_features(sample_transactions):
    features = engineer_customer_features(sample_transactions)
    assert "CustomerId" in features.columns
    assert features["frequency"].min() >= 1
    assert (features["monetary_total"] > 0).all()


def test_create_proxy_target_labels(sample_transactions):
    rfm = compute_rfm(sample_transactions)
    labeled = create_proxy_target(rfm, n_clusters=3)
    assert "default_risk" in labeled.columns
    assert set(labeled["default_risk"].unique()).issubset({0, 1})
    assert labeled["default_risk"].sum() > 0


def test_build_modeling_dataset(sample_transactions):
    dataset, metadata = build_modeling_dataset(sample_transactions, n_clusters=3)
    assert len(dataset) > 0
    assert "default_risk" in dataset.columns
    assert metadata["n_customers"] == len(dataset)
    assert len(metadata["selected_features"]) > 0


def test_select_features_by_iv(sample_transactions):
    dataset, _ = build_modeling_dataset(sample_transactions, n_clusters=3)
    selected, iv_table = select_features_by_iv(
        dataset,
        ["recency_days", "frequency", "monetary_total"],
        min_iv=0.0,
    )
    assert len(selected) >= 1
    assert "iv" in iv_table.columns


def test_save_processed_data(sample_transactions, tmp_path):
    dataset, metadata = build_modeling_dataset(sample_transactions, n_clusters=3)
    save_processed_data(dataset, metadata, tmp_path)
    assert (tmp_path / "customer_features.csv").exists()
    assert (tmp_path / "metadata.json").exists()
    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["n_customers"] == len(dataset)


def test_load_raw_missing_file():
    with pytest.raises(FileNotFoundError):
        from src.data_processing import load_raw_transactions

        load_raw_transactions(Path("nonexistent.csv"))
