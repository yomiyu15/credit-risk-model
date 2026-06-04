import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_and_evaluate_models(data_path: str):
    """
    Task 5: Prepares datasets, trains competitive models, and prints evaluation metrics.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Model-ready dataset missing at: {data_path}")

    # Load model ready records
    df = pd.read_csv(data_path)

    # Split features and our engineered high-risk target labels
    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']

    # Perform stratified split to maintain exact class proportions across training and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training subset dimensions: {X_train.shape}")
    print(f"Testing subset dimensions: {X_test.shape}\n")

    # Initialize candidate algorithms
    candidate_models = {
        "Logistic_Regression_Baseline": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest_Production": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }

    # Iterate, train, and report metrics
    for model_name, model in candidate_models.items():
        print(f"=== Training Model: {model_name} ===")
        model.fit(X_train, y_train)

        # Class predictions and continuous probability matrix generation
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculate standard Basel-aligned risk performance metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {auc:.4f}")
        print("===================================\n")


if __name__ == "__main__":
    train_and_evaluate_models(data_path="data/processed/model_ready.csv")
