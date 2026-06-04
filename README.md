# Credit Risk Probability Model for Alternative Data

End-to-end credit scoring for **Bati Bank** buy-now-pay-later partnership using Xente eCommerce transaction data. The system engineers an RFM-based proxy for default, trains classifiers with MLflow tracking, and serves risk probability, credit score, and loan terms via a containerized FastAPI service.

## Business objective

| Deliverable                  | Implementation                                          |
| ---------------------------- | ------------------------------------------------------- |
| Proxy for default (good/bad) | K-means on RFM; highest-risk cluster → `is_high_risk=1` |
| Predictive features          | Customer aggregates + WoE/IV selection                  |
| Risk probability             | sklearn classifier `predict_proba`                      |
| Credit score                 | PDO-style mapping from default probability (300–850)    |
| Loan amount & duration       | Score-band rules × monetary capacity                    |

## Credit Scoring Business Understanding

This section summarizes the regulatory and business context for Bati Bank’s buy-now-pay-later credit model, drawing on Basel II principles, alternative-data scoring practice (e.g., HKMA, World Bank), and standard scorecard development guidance.

### How does Basel II influence the need for an interpretable, well-documented model?

Basel II requires banks to hold capital in line with **measured credit risk**, not only with rules of thumb. For retail and SME portfolios, the **Internal Ratings-Based (IRB)** approach expects institutions to estimate key risk parameters—**Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)**—using models that are **sound, documented, and validated** over time.

That emphasis on risk measurement has direct modeling implications:

- **Interpretability**: Supervisors and internal risk teams must understand _why_ a customer receives a given score. Black-box outputs are harder to defend in model validation, stress testing, and fair-lending reviews.
- **Documentation**: Model purpose, data lineage, assumptions, limitations, and override policies must be recorded. Documentation is not optional paperwork—it is evidence that the bank can explain exposures to regulators and auditors.
- **Governance and monitoring**: Basel II expects ongoing **backtesting** and **performance monitoring**. If a model drifts or proxy definitions change, the bank must detect degradation and recalibrate or rebuild.
- **Pillar 2 (supervisory review)**: Even strong models face qualitative scrutiny. A well-documented, interpretable design reduces operational and reputational risk when leadership or regulators challenge automated decisions.

For Bati Bank’s partnership with an eCommerce platform, Basel II does not mandate a specific algorithm, but it does require that PD-like outputs used in underwriting and capital planning be **traceable, stable, and challengeable**—which favors transparent features, auditable score mappings, and retained experiment records (e.g., via MLflow in this project).

### Why is a proxy variable necessary without a direct “default” label, and what risks does it introduce?

Traditional credit scoring trains on a **historical default outcome** (e.g., 90+ days past due within 12 months). The Xente transaction feed provides behavioral and fraud signals, not loan performance. Without a labeled default field, supervised learning cannot target true credit loss directly.

A **proxy target** is therefore necessary: an observable label that approximates elevated credit risk when real default data is unavailable. In this project, **RFM-based segmentation** (Recency, Frequency, Monetary) defines customer groups; the segment with the weakest engagement profile is labeled high risk (`is_high_risk = 1`). This follows alternative-scoring practice where non-traditional data (transaction patterns, cash-flow proxies) supplements thin credit files (see HKMA and World Bank guidance on alternative data).

**Business risks of proxy-based prediction** include:

| Risk                       | Description                                                                                                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Concept validity**       | The proxy may measure _inactivity_ or _churn_, not willingness or ability to repay a loan. Good BNPL candidates could be misclassified if they are new or seasonal shoppers. |
| **Selection bias**         | Patterns in eCommerce data may not generalize to the loan population or economic downturns not present in the training window.                                               |
| **Fairness and inclusion** | Proxies tied to spend or frequency can disadvantage lower-volume but creditworthy customers; disparate impact must be monitored.                                             |
| **Regulatory challenge**   | Underwriters and regulators may not accept proxy PD for capital purposes without backtesting against actual defaults once loans are booked.                                  |
| **Overconfidence**         | Strong model metrics on a proxy do not prove predictive power on real default—approval policies should treat scores as **one input**, not the sole decision rule.            |

Mitigations used in this project: document the proxy definition, report Information Value (IV) for features, compare interpretable and complex models, and plan post-launch validation once BNPL performance data exists.

### Trade-offs: interpretable models (e.g., logistic regression + WoE) vs. high-performance models (e.g., gradient boosting)

| Dimension            | Interpretable (logistic regression + WoE)                                                        | High-performance (e.g., gradient boosting)                                     |
| -------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **Transparency**     | Coefficients and Weight of Evidence (WoE) bins are explainable to risk committees and customers. | Non-linear interactions are opaque; requires SHAP/LIME for local explanations. |
| **Regulatory fit**   | Aligns with classical **scorecard** development (World Bank / industry playbooks).               | Often used as challenger or secondary model after validation.                  |
| **Predictive power** | May underfit complex behavioral patterns in alternative data.                                    | Typically higher ROC-AUC and ranking of risky customers.                       |
| **Maintenance**      | Easier to recalibrate score points and policy cutoffs.                                           | Retraining and drift monitoring are more complex.                              |
| **Data needs**       | Stable binning; sensitive to sparse categories.                                                  | Handles mixed signals and interactions with less manual engineering.           |

**Practical recommendation for a regulated context:** use an **interpretable model as the primary decision narrative** (or for scorecard-style credit grades) while training **ensemble/boosted models as challengers** logged in MLflow. Select production models based on a balance of **AUC, stability, and explainability**, not AUC alone. Where boosting wins materially on validation, document supplemental explainability analysis and human-review triggers for high-exposure decisions.

For Bati Bank BNPL, logistic regression with WoE-transformed features supports Basel-style documentation; gradient boosting helps discover non-linear patterns in RFM and channel behavior—both are trained in `src/train.py` for comparison.

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

**Task 1 — Credit risk foundations**

- [Credit Scoring: Statistical Analysis (Academia Sinica)](https://www.stat.sinica.edu.tw/statscim/files/paper/48.pdf)
- [Alternative Credit Scoring of MSMEs (HKMA)](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Credit Scoring Approaches Guidelines (World Bank)](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [How to Develop a Credit Risk Model and Scorecard](https://www.linkedin.com/pulse/how-develop-credit-risk-model-scorecard-sanjay-dwivedi)
- [Credit Risk (Corporate Finance Institute)](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)

**Project & regulation**

- [Basel Framework (BIS)](https://www.bis.org/basel_framework/chapter/HTML/BaselFramework.htm)
- RFM segmentation for behavioral credit proxies
- Weight of Evidence (WoE) and Information Value (IV) for scorecard features
