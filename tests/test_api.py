"""API integration tests."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_score_without_model():
    payload = {
        "recency_days": 10,
        "frequency": 50,
        "monetary_total": 250000,
        "avg_transaction_value": 5000,
        "transaction_value_std": 1000,
        "max_transaction_value": 20000,
        "channel_diversity": 2,
        "product_category_diversity": 3,
        "fraud_rate": 0.01,
        "pay_later_share": 0.1,
        "debit_share": 0.8,
        "pricing_strategy_mean": 1,
        "tenure_days": 90,
        "transactions_per_day": 0.5,
    }
    response = client.post("/score", json=payload)
    # 503 when model not trained in CI; acceptable
    assert response.status_code in (200, 503)
