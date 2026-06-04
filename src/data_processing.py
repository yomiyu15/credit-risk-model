import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class RFMTargetEngineer(BaseEstimator, TransformerMixin):
    """
    Task 4: Extracts customer-level RFM profiles, segments users with K-Means,
    identifies the least-engaged (highest risk) cluster, and flags them as 1.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=3, random_state=self.random_state, n_init=10)
        self.high_risk_cluster_id = None

    def fit(self, X, y=None):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Consistent snapshot date formulation (max date + 1 day)
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        # Aggregate transactional records up to the CustomerId level
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        })

        # Scale behavioral values to prevent scale dominance in distance calculation
        scaled_rfm = self.scaler.fit_transform(rfm)
        self.kmeans.fit(scaled_rfm)

        # Identify the high-risk cluster center: High Recency (dormant), Low Frequency, Low Monetary
        # Score calculation in scaled space: Maximize Recency, Minimize Freq & Monetary
        cluster_centers = self.kmeans.cluster_centers_
        risk_scores = cluster_centers[:, 0] - \
            cluster_centers[:, 1] - cluster_centers[:, 2]
        self.high_risk_cluster_id = np.argmax(risk_scores)

        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        # FIXED: Kept names matching exactly what was defined during fit
        cust_rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        })

        # Assign cluster allocations and create the binary high-risk target proxy
        scaled_features = self.scaler.transform(cust_rfm)
        cust_rfm['cluster'] = self.kmeans.predict(scaled_features)
        cust_rfm['is_high_risk'] = (
            cust_rfm['cluster'] == self.high_risk_cluster_id).astype(int)

        # Join the high-risk target labels back into the main transactional dataframe
        return df.merge(cust_rfm[['is_high_risk']], on='CustomerId', how='left')


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Task 3: Production Feature Engineering Pipeline.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.freq_maps = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.scaler.fit(df[['Amount', 'Value']])

        categorical_features = ['ProviderId',
                                'ProductId', 'ProductCategory', 'ChannelId']
        for col in categorical_features:
            self.freq_maps[col] = df[col].value_counts(
                normalize=True).to_dict()

        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Temporal Extractions
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        # Aggregate Ratio Feature
        df['Amount_Value_Ratio'] = df['Amount'] / \
            (df['Value'].replace(0, 1e-5))

        # Numeric Scaling
        df[['Amount', 'Value']] = self.scaler.transform(
            df[['Amount', 'Value']])

        # Frequency Encoding
        for col in ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']:
            df[f'{col}_freq'] = df[col].map(self.freq_maps[col]).fillna(0.0)

        # Target inclusion and feature pruning
        production_features = [
            'Amount', 'Value', 'TransactionHour', 'TransactionDay',
            'TransactionMonth', 'TransactionYear', 'Amount_Value_Ratio',
            'ProviderId_freq', 'ProductId_freq', 'ProductCategory_freq', 'ChannelId_freq'
        ]

        if 'is_high_risk' in df.columns:
            production_features.append('is_high_risk')

        return df[production_features]


def execute_full_processing_pipeline(raw_input_path: str, output_processed_path: str):
    """
    Orchestrates the data pipeline: Target Engineering (Task 4) -> Feature Engineering (Task 3).
    """
    if not os.path.exists(raw_input_path):
        raise FileNotFoundError(f"Raw data file missing at: {raw_input_path}")

    print(f"Reading raw data from: {raw_input_path}...")
    df_raw = pd.read_csv(raw_input_path)

    # 1. Inject the engineered proxy target variable
    print("Engineering proxy target labels via RFM customer clustering...")
    target_engineer = RFMTargetEngineer(random_state=42)
    df_with_target = target_engineer.fit_transform(df_raw)

    # 2. Extract and transform analytical features
    print("Running feature engineering pipeline...")
    feature_pipeline = FeaturePipeline()
    df_processed = feature_pipeline.fit_transform(df_with_target)

    # Persist the final model-ready dataset
    os.makedirs(os.path.dirname(output_processed_path), exist_ok=True)
    df_processed.to_csv(output_processed_path, index=False)
    print(
        f"Successfully processed and stored model-ready dataset at: {output_processed_path}")
    print(f"Final dataset dimensions: {df_processed.shape}")
    print(f"\nTarget class distribution:")
    print(df_processed['is_high_risk'].value_counts())


if __name__ == "__main__":
    execute_full_processing_pipeline(
        raw_input_path="data/raw/data.csv",
        output_processed_path="data/processed/model_ready.csv"
    )
