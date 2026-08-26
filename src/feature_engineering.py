"""
feature_engineering.py
-----------------------
Builds "fraud-spike" velocity/burst features from the Time column, plus
whatever static per-transaction features your dataset provides.

SCHEMA FLEXIBILITY
This module is driven by config.py, not hard-coded to the ULB dataset:
  - ENTITY_ID_COL (config.py): if your dataset has a REAL entity/card/
    account ID column, set it there and this module groups by it
    directly for true per-entity velocity features.
  - If ENTITY_ID_COL is None (e.g. the bundled ULB-style dataset, which
    has no entity identifier at all), this module falls back to a
    SYNTHETIC PROXY entity ID, derived by clustering transactions on
    their numeric feature columns (KMeans). This lets the pipeline still
    demonstrate the *mechanism* of per-entity burst detection, but the
    resulting per-entity signal is NOT equivalent to a real per-card
    velocity feature -- see README "Known limitations" before trusting
    entity-level numbers from a dataset that used this fallback.
  - VELOCITY_WINDOWS_SECONDS (config.py): the rolling window sizes.
    Defaults to 1 min / 5 min / 1 hr; change these if your data has a
    different natural cadence (e.g. days instead of minutes).
  - FEATURE_COLS (config.py): set to "auto" to automatically use every
    numeric column that isn't Time/Amount/Class/entity id, or supply an
    explicit list of column names from your own dataset.

The SYSTEM-WIDE (global) burst features always use only the real Time
column, regardless of entity configuration -- they measure the true rate
of all transactions per unit time, which is a legitimate global fraud-
spike signal for any dataset with a timestamp.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

N_PSEUDO_ENTITIES = getattr(config, "N_PSEUDO_ENTITIES", 40)
VELOCITY_WINDOWS_SECONDS = getattr(
    config, "VELOCITY_WINDOWS_SECONDS", {"1min": 60, "5min": 300, "1hr": 3600}
)

RESERVED_COLS = {"Time", "Amount", "Class", "real_entity_id", "pseudo_entity", "entity_id"}


def _numeric_feature_candidates(df: pd.DataFrame) -> list:
    """All numeric columns that aren't reserved / label / id columns, and
    aren't velocity features this module itself computes (so calling
    get_feature_columns on an already-feature-engineered DataFrame
    doesn't double-count them)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        c for c in numeric_cols
        if c not in RESERVED_COLS
        and not c.startswith("entity_")
        and not c.startswith("global_")
    ]


def add_entity_ids(df: pd.DataFrame, n_clusters: int = N_PSEUDO_ENTITIES, seed: int = 42) -> pd.DataFrame:
    """
    Ensures df has an `entity_id` column used for per-entity velocity
    features. Uses the real entity column if config.ENTITY_ID_COL was
    set (renamed to `real_entity_id` by data_loader), otherwise derives
    a synthetic proxy via KMeans clustering on the numeric feature
    columns -- see module docstring for the caveat on the proxy.
    """
    df = df.copy()
    if "real_entity_id" in df.columns:
        df["entity_id"] = df["real_entity_id"]
        return df

    feature_cols = _numeric_feature_candidates(df)
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found to cluster on for the pseudo-entity "
            "fallback. Either add config.ENTITY_ID_COL pointing at a real entity "
            "column, or ensure your dataset has numeric feature columns."
        )
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=10, batch_size=1024)
    df["entity_id"] = km.fit_predict(df[feature_cols].values)
    return df


# Backwards-compatible alias (used by earlier versions / external scripts)
def add_pseudo_entity_ids(df: pd.DataFrame, n_clusters: int = N_PSEUDO_ENTITIES, seed: int = 42) -> pd.DataFrame:
    df = add_entity_ids(df, n_clusters=n_clusters, seed=seed)
    df["pseudo_entity"] = df["entity_id"]
    return df


def _rolling_count_sum(df: pd.DataFrame, group_col: str, window_seconds: float, prefix: str) -> pd.DataFrame:
    """
    For each row, computes the count and sum(Amount) of transactions in
    `group_col` within the trailing `window_seconds` (only past+current
    data -- essential for realistic real-time / no-leakage features).
    """
    out_count = np.zeros(len(df))
    out_sum = np.zeros(len(df))

    for _, g in df.groupby(group_col, sort=False):
        idx = g.index.values
        times = g["Time"].values
        amounts = g["Amount"].values

        counts = np.zeros(len(g))
        sums = np.zeros(len(g))

        left = 0
        running_sum = 0.0
        for right in range(len(g)):
            while times[right] - times[left] > window_seconds:
                running_sum -= amounts[left]
                left += 1
            running_sum += amounts[right]
            counts[right] = right - left + 1
            sums[right] = running_sum

        out_count[idx] = counts
        out_sum[idx] = sums

    df[f"{prefix}_count"] = out_count
    df[f"{prefix}_sum_amount"] = out_sum
    return df


def _ewma_zscore(series: pd.Series, span: int = 50) -> pd.Series:
    """
    Causal EWMA-based z-score: how many (EWM) standard deviations is the
    current value away from its own EWM mean, using only past+current
    values (no look-ahead).
    """
    ewm_mean = series.ewm(span=span, adjust=False).mean()
    ewm_var = series.ewm(span=span, adjust=False).var(bias=False)
    ewm_std = np.sqrt(ewm_var.clip(lower=1e-9))
    z = (series - ewm_mean) / ewm_std
    return z.fillna(0.0)


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - Per-entity rolling count/sum over the configured windows
      - EWMA z-score anomaly signal on the per-entity shortest-window count
      - System-wide (global) transaction rate over the same windows
      - EWMA z-score on the global shortest-window rate (global spike signal)

    Assumes df is sorted by Time ascending and has an `entity_id` column
    (see add_entity_ids).
    """
    df = df.sort_values("Time").reset_index(drop=True)
    windows = VELOCITY_WINDOWS_SECONDS
    shortest_label = min(windows, key=windows.get)

    # ---- Per-entity velocity features ----
    for label, secs in windows.items():
        df = _rolling_count_sum(df, "entity_id", secs, f"entity_{label}")

    df[f"entity_{shortest_label}_count_zscore"] = (
        df.groupby("entity_id")[f"entity_{shortest_label}_count"]
        .transform(lambda s: _ewma_zscore(s, span=50))
    )

    # ---- System-wide (global) velocity features ----
    df["_global"] = 0  # single group = whole system
    for label, secs in windows.items():
        df = _rolling_count_sum(df, "_global", secs, f"global_{label}")
    df = df.drop(columns=["_global"])

    df[f"global_{shortest_label}_count_zscore"] = _ewma_zscore(
        df[f"global_{shortest_label}_count"], span=50
    )

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: entity assignment (real or proxy) + velocity features,
    combined with the dataset's own static feature columns. Returns a
    DataFrame ready for train/test split (still contains Time and Class
    for splitting/labeling -- drop them from X right before model.fit).
    """
    df = add_entity_ids(df)
    df = add_velocity_features(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns the full list of model input columns: the dataset's static
    features (config.FEATURE_COLS, or auto-detected numeric columns) +
    Amount + the velocity features this module adds.
    """
    if config.FEATURE_COLS == "auto":
        static_cols = _numeric_feature_candidates(df)
        static_cols = [c for c in static_cols if c != "Amount"]
    else:
        static_cols = list(config.FEATURE_COLS)

    shortest_label = min(VELOCITY_WINDOWS_SECONDS, key=VELOCITY_WINDOWS_SECONDS.get)
    velocity_cols = []
    for label in VELOCITY_WINDOWS_SECONDS:
        velocity_cols += [f"entity_{label}_count", f"entity_{label}_sum_amount"]
    velocity_cols.append(f"entity_{shortest_label}_count_zscore")
    for label in VELOCITY_WINDOWS_SECONDS:
        velocity_cols += [f"global_{label}_count", f"global_{label}_sum_amount"]
    velocity_cols.append(f"global_{shortest_label}_count_zscore")

    return static_cols + ["Amount"] + velocity_cols


if __name__ == "__main__":
    from data_loader import load_transactions

    df = load_transactions()
    feat_df = build_feature_matrix(df)
    cols = get_feature_columns(feat_df)
    print(f"{len(cols)} feature columns: {cols}")
    print(feat_df[cols].describe().T[["mean", "std", "min", "max"]])
