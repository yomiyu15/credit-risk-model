# Credit Risk Probability Model for Alternative Data

### Bati Bank & eCommerce BNPL Partnership Implementation

---

## Credit Scoring Business Understanding

### 1. Basel II Regulatory Influence on Interpretability & Documentation

The Basel II Accord (specifically Pillar 1's Internal Ratings-Based framework) allows financial institutions to utilize internal statistical models to estimate parameters like the Probability of Default (PD). Because these calculations directly dictate the minimum regulatory capital requirements Bati Bank must maintain, risk models face strict validation scrutiny.

An interpretable and well-documented model is a regulatory necessity for several reasons:

- **Auditability & Traceability:** Regulators require the bank to prove exactly _how_ a specific risk score is derived from input data. A transparent mathematical pipeline ensures that calculations can be audited at any time.
- **Fairness & Non-Discrimination:** Transparent models allow compliance teams to verify that alternative data attributes (such as eCommerce behavior) do not serve as illegal proxies for protected status characteristics (race, gender, or religion).
- **Capital Stability:** Transparent models prevent systemic "black-box" failures during unprecedented macroeconomic shifts, as risk officers can clearly see which macroeconomic or behavioral variables are driving risk escalations.

### 2. The Necessity of a Proxy Target Variable & Associated Business Risks

In this deployment, the raw transaction dataset from our eCommerce partner contains zero historical records of loan defaults, missed payment schedules, or credit utilization lines. To train a supervised machine learning model to predict default, we must engineer a **Proxy Target Variable** ($is\_high\_risk$) using behavioral customer metrics (Recency, Frequency, and Monetary values).

While this allows us to bootstrap a model without historical credit logs, it introduces three major business risks:

- **Label Noise (Misclassification Risk):** A highly active consumer who suddenly stops purchasing might be classified by an RFM algorithm as "high-risk" (due to high recency), when in reality they simply migrated to a competitor. Misclassifying these users leads to missed revenue.
- **Adverse Selection:** If our proxy fails to capture subtle financial distress patterns, the bank might inadvertently grant high credit limits to consumers on the verge of default, leading to an elevated non-performing loan (NPL) ratio.
- **Behavioral Drift:** eCommerce behaviors adapt rapidly to marketing cycles, seasonal sales, and interface updates. An RFM proxy can change abruptly, decoupling it from actual financial repayment capacity.

### 3. Methodology Trade-Offs: Interpretable Traditional vs. High-Performance ML

| Dimension                 | Traditional Scorecard (Logistic Regression + WoE)                                                                                                          | Advanced Ensembles (XGBoost / Random Forest)                                                                                                  |
| :------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| **Interpretability**      | **Extremely High.** Features are binned using Weight of Evidence (WoE). Highly transparent; can be directly converted into a point-based credit scorecard. | **Low to Moderate.** Relies on post-hoc explainability frameworks (such as SHAP or LIME) that approximate feature contributions.              |
| **Regulatory Compliance** | **Seamless.** Fully maps to Basel II standards for explicit mathematical transparency and clear audit trails.                                              | **Challenging.** Requires intense validation, model stability proofs, and backtesting to clear internal bank risk committees.                 |
| **Predictive Power**      | **Moderate.** Struggles to capture non-linear combinations or complex feature interactions without manual interaction engineering.                         | **High.** Automatically uncovers and leverages multi-dimensional interactions and subtle behavioral nuances in alternative data.              |
| **Operational Stability** | **High.** Binning categorical attributes via WoE limits the impact of outliers and ensures the model is resilient to data noise.                           | **Low to Moderate.** Susceptible to overfitting on noisy or uncurated alternative data streams if not constrained with strict regularization. |

---
