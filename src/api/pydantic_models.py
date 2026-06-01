"""Request and response schemas for the credit scoring API."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Customer-level features (same as training pipeline output)."""

    recency_days: float = Field(..., ge=0, description="Days since last transaction")
    frequency: float = Field(..., ge=0, description="Total transaction count")
    monetary_total: float = Field(..., ge=0, description="Sum of transaction values")
    avg_transaction_value: float = Field(..., ge=0)
    transaction_value_std: float = Field(0, ge=0)
    max_transaction_value: float = Field(..., ge=0)
    channel_diversity: float = Field(1, ge=0)
    product_category_diversity: float = Field(1, ge=0)
    fraud_rate: float = Field(0, ge=0, le=1)
    pay_later_share: float = Field(0, ge=0, le=1)
    debit_share: float = Field(0, ge=0, le=1)
    pricing_strategy_mean: float = Field(0)
    tenure_days: float = Field(1, ge=1)
    transactions_per_day: float = Field(0, ge=0)


class ScoreResponse(BaseModel):
    default_probability: float
    credit_score: int
    risk_band: str
    recommended_loan_amount: float
    recommended_duration_months: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
