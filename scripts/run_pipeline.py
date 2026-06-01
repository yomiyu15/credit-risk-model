#!/usr/bin/env python3
"""End-to-end pipeline: process -> train -> smoke-test API."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import run_processing_pipeline
from src.train import train_all_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full credit risk pipeline")
    parser.add_argument(
        "--input",
        default="data/raw/training.csv",
        help="Raw transaction CSV path",
    )
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    raw = Path(args.input)
    if not raw.exists():
        logger.error(
            "Place Xente training.csv at %s (see README).", raw
        )
        sys.exit(1)

    logger.info("Step 1/2: Feature engineering and proxy target")
    run_processing_pipeline(raw, args.processed)

    logger.info("Step 2/2: Model training with MLflow")
    artifact = train_all_models(args.processed, args.models)
    logger.info("Done. Best model: %s", artifact["model_name"])


if __name__ == "__main__":
    main()
