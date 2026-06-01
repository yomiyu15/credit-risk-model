"""
Inference: risk probability, credit score, and loan terms.

Credit score uses a PDO-style linear transformation of log-odds (Basel-friendly,
auditable mapping from probability to score).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Score calibration: anchor point (score at 50% odds) and points to double odds
BASE_SCORE = 600
PDO = 20
ODDS_AT_BASE = 1.0


def _load_metadata(model_dir: Path) -> dict[str, Any]:
    meta_path = model_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_model(model_dir: str | Path = "models"):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "credit_risk_model.joblib")
    metadata = _load_metadata(model_dir)
    return model, metadata


def probability_to_credit_score(probability: float) -> int:
    """
    Map default probability to credit score (300–850).

    Lower default probability => higher credit score.
    """
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    # Invert: default odds -> score (good customer has low p, high score)
    good_odds = (1 - p) / p
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(ODDS_AT_BASE)
    score = offset + factor * np.log(good_odds)
    return int(np.clip(round(score), 300, 850))


def recommend_loan_terms(
    credit_score: int,
    monetary_total: float,
    avg_transaction_value: float,
) -> dict[str, float]:
    """
    Suggest loan amount and duration from score band and spending capacity.

    Amount capped as a fraction of observed monetary activity; duration
    lengthens for stronger scores.
    """
    capacity = max(monetary_total, avg_transaction_value * 10, 1000.0)

    if credit_score >= 750:
        amount_ratio, duration_months = 0.5, 24
    elif credit_score >= 650:
        amount_ratio, duration_months = 0.35, 18
    elif credit_score >= 550:
        amount_ratio, duration_months = 0.2, 12
    else:
        amount_ratio, duration_months = 0.1, 6

    recommended_amount = round(capacity * amount_ratio, 2)
    return {
        "recommended_loan_amount": recommended_amount,
        "recommended_duration_months": float(duration_months),
    }


def predict_customer(
    features: dict[str, float] | pd.Series,
    model=None,
    metadata: dict[str, Any] | None = None,
    model_dir: str | Path = "models",
) -> dict[str, Any]:
    """Score a single customer from engineered feature dict."""
    if model is None or metadata is None:
        model, metadata = load_model(model_dir)

    feature_cols = metadata["feature_columns"]
    row = pd.DataFrame([{k: features.get(k, 0.0) for k in feature_cols}])

    prob = float(model.predict_proba(row[feature_cols])[:, 1][0])
    score = probability_to_credit_score(prob)
    loan = recommend_loan_terms(
        score,
        float(features.get("monetary_total", 0)),
        float(features.get("avg_transaction_value", 0)),
    )

    risk_band = (
        "high" if prob >= 0.5 else "medium" if prob >= 0.3 else "low"
    )

    return {
        "default_probability": round(prob, 6),
        "credit_score": score,
        "risk_band": risk_band,
        **loan,
    }


def predict_batch(
    df: pd.DataFrame,
    model=None,
    metadata: dict[str, Any] | None = None,
    model_dir: str | Path = "models",
) -> pd.DataFrame:
    """Score multiple customers."""
    if model is None or metadata is None:
        model, metadata = load_model(model_dir)

    feature_cols = metadata["feature_columns"]
    probs = model.predict_proba(df[feature_cols])[:, 1]

    results = []
    for i, prob in enumerate(probs):
        features = df.iloc[i].to_dict()
        score = probability_to_credit_score(prob)
        loan = recommend_loan_terms(
            score,
            float(features.get("monetary_total", 0)),
            float(features.get("avg_transaction_value", 0)),
        )
        results.append(
            {
                "default_probability": round(float(prob), 6),
                "credit_score": score,
                **loan,
            }
        )
    return pd.DataFrame(results)
