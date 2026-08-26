"""
train.py
--------
Trains the fraud-spike classifier and a simple statistical baseline.

KEY DESIGN CHOICE: TIME-BASED SPLIT (not random)
Fraud detection data is inherently sequential: fraud patterns drift over
time, and in production you only ever have the past to predict the
future. A random split would leak future information into training
(e.g. a burst feature computed using transactions that happen "after"
a training row, seen via a temporally-adjacent test row) and would
give an overly optimistic, unrealistic estimate of real-world
performance. We therefore sort by Time and train on the first 70%,
testing on the strictly-later last 30%.

Models:
  1. Primary: XGBoost classifier with scale_pos_weight to handle the
     extreme class imbalance (~0.17% fraud).
  2. Baseline: a simple EWMA/CUSUM statistical spike detector that only
     looks at the global_1min_count_zscore feature -- no ML, just a
     threshold on the statistical anomaly signal. Used as a sanity-check
     comparison point in evaluate.py.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from data_loader import load_transactions
from feature_engineering import build_feature_matrix, get_feature_columns, VELOCITY_WINDOWS_SECONDS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TRAIN_FRACTION = 0.70
RANDOM_SEED = 42
_SHORTEST_WINDOW_LABEL = min(VELOCITY_WINDOWS_SECONDS, key=VELOCITY_WINDOWS_SECONDS.get)


def time_based_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """
    Splits a Time-sorted DataFrame into train (earliest `train_fraction`)
    and test (remaining, strictly later) sets. No shuffling.
    """
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * train_fraction)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def train_xgb_classifier(X_train, y_train, seed: int = RANDOM_SEED) -> XGBClassifier:
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


class EwmaCusumBaseline:
    """
    Simple, non-ML statistical baseline detector.

    Flags a transaction as a "spike" if EITHER:
      - its per-entity EWMA z-score (entity_1min_count_zscore) exceeds
        `z_threshold`, OR
      - a CUSUM statistic accumulated on the global 1-minute count
        z-score exceeds `cusum_threshold` (detects sustained, slow-
        building drift as well as sharp single-point spikes).

    This exists purely as an interpretable, zero-training-cost sanity
    baseline to compare the ML model against in evaluate.py -- it is not
    intended to beat the ML model, but to show the ML model is adding
    real value over a naive statistical rule.
    """

    def __init__(self, z_threshold: float = 3.0, cusum_threshold: float = 5.0, drift: float = 0.5):
        self.z_threshold = z_threshold
        self.cusum_threshold = cusum_threshold
        self.drift = drift

    def score(self, df: pd.DataFrame) -> np.ndarray:
        entity_z = df[f"entity_{_SHORTEST_WINDOW_LABEL}_count_zscore"].values
        global_z = df[f"global_{_SHORTEST_WINDOW_LABEL}_count_zscore"].values

        # CUSUM on global z-score (one-sided, upward drift detector)
        cusum = np.zeros(len(global_z))
        running = 0.0
        for i, z in enumerate(global_z):
            running = max(0.0, running + z - self.drift)
            cusum[i] = running

        # Combine both signals into a single continuous "score" in [0, ~1]
        # via a smooth max of the two normalized signals, for a fair
        # comparison against the ML model's probability output.
        entity_signal = 1 / (1 + np.exp(-(entity_z - self.z_threshold)))
        cusum_signal = 1 / (1 + np.exp(-(cusum - self.cusum_threshold)))
        combined = np.maximum(entity_signal, cusum_signal)
        return combined

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.score(df) >= threshold).astype(int)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_transactions()
    source = df.attrs.get("source", "unknown")

    print("Building features (pseudo-entity clustering + velocity features)...")
    feat_df = build_feature_matrix(df)
    feature_cols = get_feature_columns(feat_df)

    print(f"Splitting by Time (train={TRAIN_FRACTION:.0%}, test={1-TRAIN_FRACTION:.0%})...")
    train_df, test_df = time_based_split(feat_df, TRAIN_FRACTION)
    print(f"  Train: {len(train_df):,} rows, {train_df['Class'].sum()} frauds")
    print(f"  Test:  {len(test_df):,} rows, {test_df['Class'].sum()} frauds")

    X_train, y_train = train_df[feature_cols], train_df["Class"]
    X_test, y_test = test_df[feature_cols], test_df["Class"]

    print("Training XGBoost classifier (scale_pos_weight for imbalance)...")
    model = train_xgb_classifier(X_train, y_train)

    print("Building EWMA/CUSUM statistical baseline...")
    baseline = EwmaCusumBaseline()

    # Persist artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_fraud_spike_model.joblib"))
    joblib.dump(baseline, os.path.join(MODELS_DIR, "ewma_cusum_baseline.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.joblib"))

    # Persist the split test set (with features) so evaluate.py / app.py
    # don't need to redo feature engineering + clustering.
    test_df.to_parquet(os.path.join(MODELS_DIR, "test_set_with_features.parquet"))
    train_df.to_parquet(os.path.join(MODELS_DIR, "train_set_with_features.parquet"))

    meta = {
        "data_source": source,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_train_frauds": int(train_df["Class"].sum()),
        "n_test_frauds": int(test_df["Class"].sum()),
        "feature_columns": feature_cols,
        "train_fraction": TRAIN_FRACTION,
    }
    with open(os.path.join(MODELS_DIR, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model, baseline, and metadata to {MODELS_DIR}/")
    print(f"Data source used for this run: {source}")


if __name__ == "__main__":
    main()
