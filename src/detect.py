"""
detect.py
---------
Real-time-style inference: given a new transaction plus recent history
for its entity (and, ideally, recent system-wide history), computes the
fraud-spike velocity features on the fly and returns a score + flag
decision using the trained model and the cost-optimal threshold.

This module does NOT retrain anything and does NOT use future data --
it only ever looks backward from the new transaction's timestamp,
mirroring how the system would behave in production.

SCHEMA FLEXIBILITY
Window sizes come from config.VELOCITY_WINDOWS_SECONDS, so this module
automatically matches whatever windows train.py / feature_engineering.py
were configured with -- no hard-coded "1min"/"5min"/"1hr" here.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from feature_engineering import _ewma_zscore, VELOCITY_WINDOWS_SECONDS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

_SHORTEST_LABEL = min(VELOCITY_WINDOWS_SECONDS, key=VELOCITY_WINDOWS_SECONDS.get)
_SHORTEST_SECS = VELOCITY_WINDOWS_SECONDS[_SHORTEST_LABEL]
_LONGEST_SECS = max(VELOCITY_WINDOWS_SECONDS.values())


class FraudSpikeDetector:
    """
    Loads the trained model + cost-optimal threshold once, then exposes
    `score_transaction()` for repeated real-time-style scoring calls.
    """

    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, "xgb_fraud_spike_model.joblib"))
        self.feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
        self.threshold = self._load_cost_optimal_threshold()

    @staticmethod
    def _load_cost_optimal_threshold(default: float = 0.5) -> float:
        report_path = os.path.join(RESULTS_DIR, "metrics_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            return report["cost_model"]["cost_optimal_threshold"]
        return default

    def compute_velocity_features(
        self,
        new_txn: dict,
        entity_history: pd.DataFrame,
        global_history: pd.DataFrame,
    ) -> dict:
        """
        new_txn: dict with keys Time, Amount, plus this dataset's static
            feature columns, plus entity_id
        entity_history: past transactions (Time, Amount) for this same
            entity, sorted ascending, strictly before new_txn['Time']
        global_history: past transactions (Time, Amount) system-wide,
            sorted ascending, strictly before new_txn['Time']

        Returns a dict of the velocity features needed by the model,
        computed causally (only using data at/before the new txn time),
        using whatever windows are configured in config.VELOCITY_WINDOWS_SECONDS.
        """
        t_new = new_txn["Time"]
        feats = {}

        def rolling_count_sum(history_df, label, secs):
            window_df = history_df[history_df["Time"] >= t_new - secs]
            count = len(window_df) + 1  # +1 for the new transaction itself
            total = window_df["Amount"].sum() + new_txn["Amount"]
            return count, total

        def shortest_window_zscore(history_df, shortest_count):
            """
            Builds a short bucketed count series (trailing longest window,
            bucketed by the shortest window size) ending at the new
            transaction, then takes the causal EWMA z-score of the final
            (most recent) bucket -- a real-time approximation of the same
            EWMA z-score feature_engineering.py computes in batch.
            """
            bucket_counts = []
            if len(history_df) > 0:
                for secs_ago in range(0, _LONGEST_SECS, _SHORTEST_SECS):
                    lo = t_new - secs_ago - _SHORTEST_SECS
                    hi = t_new - secs_ago
                    c = ((history_df["Time"] >= lo) & (history_df["Time"] < hi)).sum()
                    bucket_counts.append(c)
                bucket_counts = list(reversed(bucket_counts))
            bucket_counts.append(shortest_count)
            z_series = _ewma_zscore(pd.Series(bucket_counts), span=50)
            return float(z_series.iloc[-1])

        # --- Entity-level rolling count/sum over each configured window ---
        entity_shortest_count = None
        for label, secs in VELOCITY_WINDOWS_SECONDS.items():
            count, total = rolling_count_sum(entity_history, label, secs)
            feats[f"entity_{label}_count"] = count
            feats[f"entity_{label}_sum_amount"] = total
            if label == _SHORTEST_LABEL:
                entity_shortest_count = count
        feats[f"entity_{_SHORTEST_LABEL}_count_zscore"] = shortest_window_zscore(
            entity_history, entity_shortest_count
        )

        # --- Global-level rolling count/sum over each configured window ---
        global_shortest_count = None
        for label, secs in VELOCITY_WINDOWS_SECONDS.items():
            count, total = rolling_count_sum(global_history, label, secs)
            feats[f"global_{label}_count"] = count
            feats[f"global_{label}_sum_amount"] = total
            if label == _SHORTEST_LABEL:
                global_shortest_count = count
        feats[f"global_{_SHORTEST_LABEL}_count_zscore"] = shortest_window_zscore(
            global_history, global_shortest_count
        )

        return feats

    def score_transaction(
        self,
        new_txn: dict,
        entity_history: pd.DataFrame,
        global_history: pd.DataFrame,
    ) -> dict:
        """
        Returns:
          {
            "fraud_spike_score": float in [0, 1],
            "flag": bool,
            "threshold_used": float,
            "velocity_features": {...},
          }
        """
        velocity_feats = self.compute_velocity_features(new_txn, entity_history, global_history)

        row = {**new_txn, **velocity_feats}
        X = pd.DataFrame([row])[self.feature_cols]

        score = float(self.model.predict_proba(X)[:, 1][0])
        flag = score >= self.threshold

        return {
            "fraud_spike_score": score,
            "flag": bool(flag),
            "threshold_used": self.threshold,
            "velocity_features": velocity_feats,
        }


if __name__ == "__main__":
    # Minimal smoke test using the saved test set (treats each row's own
    # preceding rows as its "history", exactly as feature_engineering
    # computed them, just to confirm the module runs end to end).
    test_df = pd.read_parquet(os.path.join(MODELS_DIR, "test_set_with_features.parquet"))
    detector = FraudSpikeDetector()

    sample = test_df.iloc[100]
    entity_hist = test_df[
        (test_df["entity_id"] == sample["entity_id"]) & (test_df["Time"] < sample["Time"])
    ][["Time", "Amount"]]
    global_hist = test_df[test_df["Time"] < sample["Time"]][["Time", "Amount"]]

    static_feature_cols = [
        c for c in detector.feature_cols
        if not c.startswith("entity_") and not c.startswith("global_") and c != "Amount"
    ]
    new_txn = {c: sample[c] for c in static_feature_cols}
    new_txn["Time"] = sample["Time"]
    new_txn["Amount"] = sample["Amount"]
    new_txn["entity_id"] = sample["entity_id"]

    result = detector.score_transaction(new_txn, entity_hist, global_hist)
    print(json.dumps(result, indent=2, default=float))
