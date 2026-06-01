# Credit Risk Model — Final Report (Bati Bank × Xente)

## Executive summary

Bati Bank needs a real-time credit scoring service for buy-now-pay-later applicants sourced from eCommerce behavioral data. Because the transaction feed has **no historical default labels**, we define a **regulatory-documented proxy** using RFM segmentation, train classifiers on engineered features, and deploy a REST API that returns default probability, credit score, and recommended loan terms.

---

## 1. Proxy variable design

### Problem

Credit models require a binary outcome (default vs. non-default). Xente data only provides transactions and a fraud flag—not loan performance.

### Approach: RFM + clustering

| Metric | Definition | Risk intuition |
|--------|------------|----------------|
| **Recency** | Days since last transaction | High recency → disengaged / unstable |
| **Frequency** | Transaction count | Low frequency → thin file |
| **Monetary** | Sum of transaction values | Low spend → limited capacity |

We cluster customers on standardized RFM (K-means, k=4) and assign **high risk** to the cluster with the worst profile (high recency, low frequency, low monetary).

### Business risks of the proxy

- Proxy measures **behavioral churn/inactivity**, not true loan default—approval policies must treat scores as **screening**, not sole decision basis.
- Fraud labels are excluded as the primary target to avoid conflating fraud with credit risk (fraud rate retained as a feature).
- Proxy class balance and cluster stability should be monitored monthly.

---

## 2. Feature engineering

Per-customer aggregates include:

- RFM core metrics
- Average / std / max transaction value
- Channel and product diversity
- Pay-later channel share, debit share
- Tenure and transactions per day

**WoE and Information Value (IV)** rank predictors; features with IV ≥ 0.02 are retained (weak predictors dropped for parsimony).

---

## 3. Model development

| Model | Rationale |
|-------|-----------|
| Logistic regression | Interpretable coefficients; Basel-friendly |
| Random forest | Non-linear interactions |
| Gradient boosting | Strong AUC on tabular data |

- **Preprocessing**: StandardScaler in sklearn `Pipeline`
- **Imbalance**: SMOTE on training folds
- **Selection**: 5-fold stratified CV, optimize ROC-AUC
- **Tracking**: MLflow experiment `credit-risk-bati-bank`

Report metrics: ROC-AUC, average precision, F1, precision, recall.

---

## 4. Credit score and loan terms

**Credit score** (300–850): PDO mapping on *good* odds \((1-p)/p\) with base score 600 and PDO=20.

**Loan recommendation**:

- Amount = capacity × band ratio (capacity from monetary totals)
- Duration = 6–24 months by score band

---

## 5. Deployment

- **FastAPI** `/score` — real-time scoring
- **Docker** — reproducible serving image
- **GitHub Actions** — pytest + image build on each push

---

## 6. Demonstration checklist

- [ ] Run `scripts/run_pipeline.py` on training.csv
- [ ] Review MLflow runs (`mlflow ui`)
- [ ] Call `/docs` and score a sample customer
- [ ] Attach ROC curves and IV table from notebook

---

*This report is intended for Bati Bank leadership and risk teams as a self-contained methodology artifact.*
