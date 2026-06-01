"""
Model training with MLflow experiment tracking.

Trains and compares interpretable (logistic regression) and ensemble models,
selects the best by ROC-AUC, and persists artifacts for serving.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_processing import MODEL_FEATURE_COLUMNS, RANDOM_STATE

logger = logging.getLogger(__name__)

TARGET_COL = "default_risk"
MODEL_DIR = Path("models")
MLFLOW_EXPERIMENT = "credit-risk-bati-bank"


def load_processed_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    csv_path = path / "customer_features.csv" if path.is_dir() else path
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed data not found: {csv_path}")
    return pd.read_csv(csv_path)


def _get_feature_columns(df: pd.DataFrame, metadata_path: Path | None = None) -> list[str]:
    if metadata_path and metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("selected_features", MODEL_FEATURE_COLUMNS)
    return [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("num", StandardScaler(), feature_cols)],
        remainder="drop",
    )


def get_model_candidates() -> dict[str, Any]:
    """Models aligned with Basel II interpretability vs performance trade-off."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10, None],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "clf__n_estimators": [100, 150],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
            },
        },
    }


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    feature_cols: list[str],
    use_smote: bool = True,
) -> tuple[Pipeline, dict[str, float], Any]:
    candidates = get_model_candidates()
    if model_name not in candidates:
        raise ValueError(f"Unknown model: {model_name}")

    spec = candidates[model_name]
    preprocessor = build_preprocessor(feature_cols)

    steps = [
        ("preprocess", preprocessor),
        ("clf", spec["estimator"]),
    ]
    base_pipeline: Pipeline | ImbPipeline = Pipeline(steps)

    if use_smote:
        pipeline: Pipeline | ImbPipeline = ImbPipeline(
            [
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", spec["estimator"]),
            ]
        )
        # GridSearch param names must match final estimator step
        param_grid = spec["param_grid"]
    else:
        pipeline = base_pipeline
        param_grid = spec["param_grid"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train[feature_cols], y_train)

    best = search.best_estimator_
    y_prob = best.predict_proba(X_val[feature_cols])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate_model(y_val.values, y_prob, y_pred)
    metrics["best_cv_auc"] = float(search.best_score_)

    return best, metrics, search.best_params_


def train_all_models(
    processed_dir: str | Path = "data/processed",
    model_output_dir: str | Path = "models",
    use_smote: bool = True,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train all model families, log to MLflow, save best overall."""
    processed_dir = Path(processed_dir)
    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_data(processed_dir)
    metadata_path = processed_dir / "metadata.json"
    feature_cols = _get_feature_columns(df, metadata_path)

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    results: dict[str, Any] = {}
    best_name = None
    best_auc = -1.0
    best_model = None

    for model_name in get_model_candidates():
        logger.info("Training %s...", model_name)
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("features", feature_cols)
            mlflow.log_param("use_smote", use_smote)

            model, metrics, best_params = train_single_model(
                X_train, y_train, X_val, y_val, model_name, feature_cols, use_smote=use_smote
            )

            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.sklearn.log_model(model, artifact_path="model")

            results[model_name] = {"metrics": metrics, "params": best_params}

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = model_name
                best_model = model

    assert best_model is not None and best_name is not None

    artifact = {
        "model_name": best_name,
        "feature_columns": feature_cols,
        "metrics": results[best_name]["metrics"],
        "all_results": results,
    }

    joblib.dump(best_model, model_output_dir / "credit_risk_model.joblib")
    with open(model_output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    logger.info("Best model: %s (ROC-AUC=%.4f)", best_name, best_auc)
    return artifact


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="Train credit risk models")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--output", default="models")
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()
    train_all_models(args.data, args.output, use_smote=not args.no_smote)
