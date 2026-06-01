"""Tests for scoring and loan recommendation logic."""

import numpy as np
import pytest

from src.predict import (
    probability_to_credit_score,
    recommend_loan_terms,
)


def test_credit_score_monotonic():
    low_risk = probability_to_credit_score(0.05)
    high_risk = probability_to_credit_score(0.8)
    assert low_risk > high_risk
    assert 300 <= high_risk <= 850
    assert 300 <= low_risk <= 850


def test_recommend_loan_terms_bands():
    good = recommend_loan_terms(780, 100000, 5000)
    poor = recommend_loan_terms(450, 100000, 5000)
    assert good["recommended_loan_amount"] > poor["recommended_loan_amount"]
    assert good["recommended_duration_months"] > poor["recommended_duration_months"]


@pytest.mark.parametrize("prob", [0.01, 0.5, 0.99])
def test_score_bounds(prob):
    score = probability_to_credit_score(prob)
    assert 300 <= score <= 850
