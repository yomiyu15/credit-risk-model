"""
FastAPI service for real-time credit risk scoring.

Basel II alignment: documented inputs, interpretable base models,
probability outputs, and auditable score mapping.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import CustomerFeatures, HealthResponse, ScoreResponse
from src.predict import load_model, predict_customer

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

_state: dict = {"model": None, "metadata": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    model_path = MODEL_DIR / "credit_risk_model.joblib"
    if model_path.exists():
        try:
            model, metadata = load_model(MODEL_DIR)
            _state["model"] = model
            _state["metadata"] = metadata
            logger.info("Loaded model: %s", metadata.get("model_name"))
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
    yield
    _state.clear()


app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Credit scoring from eCommerce behavioral (RFM) features",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
    )


@app.post("/score", response_model=ScoreResponse)
def score_customer(features: CustomerFeatures):
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model and place artifacts in models/",
        )
    result = predict_customer(
        features.model_dump(),
        model=_state["model"],
        metadata=_state["metadata"],
    )
    return ScoreResponse(**result)
