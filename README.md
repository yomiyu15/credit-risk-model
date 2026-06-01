# Credit Risk Probability Model for Alternative Data

End-to-end credit scoring for **Bati Bank** buy-now-pay-later partnership using Xente eCommerce transaction data. The system engineers an RFM-based proxy for default, trains classifiers with MLflow tracking, and serves risk probability, credit score, and loan terms via a containerized FastAPI service.

## Business objective

| Deliverable | Implementation |
|-------------|----------------|
| Proxy for default (good/bad) | K-means on RFM; highest-risk cluster → `default_risk=1` |
| Predictive features | Customer aggregates + WoE/IV selection |
| Risk probability | sklearn classifier `predict_proba` |
| Credit score | PDO-style mapping from default probability (300–850) |
| Loan amount & duration | Score-band rules × monetary capacity |

## Project structure

```
credit-risk-model/
├── .github/workflows/ci.yml
├── data/raw/              # Place training.csv here (gitignored)
├── data/processed/        # Generated customer features
├── notebooks/eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
├── tests/
├── models/                # Trained model artifacts
├── Dockerfile
└── docker-compose.yml
```

## Quick start

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Data

Download **training.csv** from the [Xente challenge](https://zindi.africa/competitions/xente-fraud-detection-challenge/data) (or your course portal) and save as:

```
data/raw/training.csv
```

Do not commit raw data to public repositories (competition terms).

### 3. Run pipeline

```bash
python scripts/run_pipeline.py --input data/raw/training.csv
```

Or step by step:

```bash
python -m src.data_processing --input data/raw/training.csv --output data/processed
python -m src.train --data data/processed --output models
```

### 4. API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d "{\"recency_days\": 5, \"frequency\": 120, \"monetary_total\": 500000, \"avg_transaction_value\": 4000, \"max_transaction_value\": 50000, \"transaction_value_std\": 2000, \"channel_diversity\": 3, \"product_category_diversity\": 4, \"fraud_rate\": 0.02, \"pay_later_share\": 0.15, \"debit_share\": 0.7, \"pricing_strategy_mean\": 1, \"tenure_days\": 120, \"transactions_per_day\": 1.0}"
```

### 5. Docker

```bash
docker compose up --build
```

API: http://localhost:8000/docs  
MLflow UI (optional): http://localhost:5000

## Basel II alignment

- **Interpretability**: Logistic regression included; feature list documented in `metadata.json`
- **Documentation**: Proxy rationale and IV summary in processed metadata; see `docs/REPORT.md`
- **Monitoring**: MLflow metrics (ROC-AUC, precision, recall); log predictions in production

## Testing

```bash
pytest
```

## Team

Kerod · Mahbubah · Feven

## References

- [Basel II overview (BIS)](https://www.bis.org/basel_framework/chapter/HTML/BaselFramework.htm)
- RFM segmentation for behavioral credit proxies
- WoE / IV for feature strength in scorecards
