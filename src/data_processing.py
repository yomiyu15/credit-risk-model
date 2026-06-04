import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Task 3: Production Feature Engineering Pipeline.
    Encapsulates scaling, categorical frequency mapping, and temporal feature extraction.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.freq_maps = {}

    def fit(self, X, y=None):
        # Prevent side effects on raw dataframes
        df = X.copy()

        # Fit our scaling matrix on numerical continuous attributes
        self.scaler.fit(df[['Amount', 'Value']])

        # Build Frequency Encoding dictionaries for high-cardinality categorical columns
        categorical_features = ['ProviderId',
                                'ProductId', 'ProductCategory', 'ChannelId']
        for col in categorical_features:
            # Map categories to their relative frequency probability within training data
            self.freq_maps[col] = df[col].value_counts(
                normalize=True).to_dict()

        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # 1. Temporal Component Extractions
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        # 2. Advanced Aggregate Features (Derived at Transaction Level)
        df['Amount_Value_Ratio'] = df['Amount'] / \
            (df['Value'].replace(0, 1e-5))

        # 3. Numeric Standardization Transform
        df[['Amount', 'Value']] = self.scaler.transform(
            df[['Amount', 'Value']])

        # 4. Apply Categorical Frequency Encoding with fallback defaults for out-of-vocabulary terms
        for col in ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']:
            df[f'{col}_freq'] = df[col].map(self.freq_maps[col]).fillna(0.0)

        # 5. Filter down to production-only model features
        production_features = [
            'Amount', 'Value', 'TransactionHour', 'TransactionDay',
            'TransactionMonth', 'TransactionYear', 'Amount_Value_Ratio',
            'ProviderId_freq', 'ProductId_freq', 'ProductCategory_freq', 'ChannelId_freq'
        ]

        # Preserve original contextual identifiers for downstream merging or debugging
        identifiers = ['TransactionId', 'CustomerId']
        for id_col in identifiers:
            if id_col in df.columns:
                production_features.insert(0, id_col)

        # Preserve existing target variable if present during training phase
        if 'FraudResult' in df.columns:
            production_features.append('FraudResult')

        return df[production_features]


def execute_feature_engineering(raw_input_path: str, output_processed_path: str):
    """
    Loads raw Xente transaction data and executes the FeaturePipeline transform.
    """
    if not os.path.exists(raw_input_path):
        raise FileNotFoundError(f"Raw data file missing at: {raw_input_path}")

    print(f"Reading raw data from: {raw_input_path}...")
    df_raw = pd.read_csv(raw_input_path)

    # Initialize and fit the pipeline object
    pipeline = FeaturePipeline()
    df_processed = pipeline.fit_transform(df_raw)

    # Ensure processed output storage directories exist
    os.makedirs(os.path.dirname(output_processed_path), exist_ok=True)
    df_processed.to_csv(output_processed_path, index=False)
    print(
        f"Successfully engineered and stored dataset at: {output_processed_path}")
    print(f"Processed features shape: {df_processed.shape}")


if __name__ == "__main__":
    execute_feature_engineering(
        raw_input_path="data/raw/data.csv",
        output_processed_path="data/processed/features_engineered.csv"
    )
