"""
test_pipeline.py
-----------------
Basic sanity tests for the fraud-spike detector pipeline. Run with:
    pytest tests/test_pipeline.py -v

These are not exhaustive unit tests -- they check that each pipeline
stage runs, produces the expected shape/columns, and that core
invariants hold (e.g. no look-ahead leakage in the time-based split,
velocity features are non-negative, causal EWMA z-score doesn't use
future data).
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import generate_synthetic_fallback
from feature_engineering import (
    add_pseudo_entity_ids, add_velocity_features, build_feature_matrix,
    get_feature_columns, _ewma_zscore,
)
from train import time_based_split, EwmaCusumBaseline


@pytest.fixture(scope="module")
def small_df():
    return generate_synthetic_fallback(n_rows=4000, fraud_rate=0.01, seed=1)


def test_synthetic_data_shape(small_df):
    expected_cols = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    assert expected_cols.issubset(set(small_df.columns))
    assert len(small_df) > 0
    assert small_df["Class"].isin([0, 1]).all()
    assert small_df["Time"].is_monotonic_increasing


def test_pseudo_entity_assignment(small_df):
    df = add_pseudo_entity_ids(small_df, n_clusters=10)
    assert "pseudo_entity" in df.columns
    assert df["pseudo_entity"].nunique() <= 10
    assert df["pseudo_entity"].nunique() > 1


def test_velocity_features_non_negative(small_df):
    df = add_pseudo_entity_ids(small_df, n_clusters=10)
    df = add_velocity_features(df)
    count_cols = [c for c in df.columns if c.endswith("_count")]
    sum_cols = [c for c in df.columns if c.endswith("_sum_amount")]
    for c in count_cols:
        assert (df[c] >= 1).all(), f"{c} should be >= 1 (includes current txn)"
    for c in sum_cols:
        assert (df[c] >= 0).all(), f"{c} should be non-negative"


def test_1min_count_leq_5min_leq_1hr(small_df):
    """Sanity: a shorter window can never contain more transactions than a longer one."""
    df = add_pseudo_entity_ids(small_df, n_clusters=10)
    df = add_velocity_features(df)
    assert (df["entity_1min_count"] <= df["entity_5min_count"]).all()
    assert (df["entity_5min_count"] <= df["entity_1hr_count"]).all()
    assert (df["global_1min_count"] <= df["global_5min_count"]).all()
    assert (df["global_5min_count"] <= df["global_1hr_count"]).all()


def test_ewma_zscore_is_causal():
    """
    The EWMA z-score at position i must be identical whether or not
    future values (after i) exist in the series -- i.e. no look-ahead.
    """
    s = pd.Series(np.concatenate([np.ones(20), np.array([50.0]), np.ones(20)]))
    z_full = _ewma_zscore(s, span=10)

    s_truncated = s.iloc[:21]  # up to and including the spike at index 20
    z_truncated = _ewma_zscore(s_truncated, span=10)

    pd.testing.assert_series_equal(
        z_full.iloc[:21].reset_index(drop=True),
        z_truncated.reset_index(drop=True),
        check_names=False,
    )


def test_time_based_split_no_overlap_and_ordering(small_df):
    df = add_pseudo_entity_ids(small_df, n_clusters=10)
    df = add_velocity_features(df)
    train_df, test_df = time_based_split(df, train_fraction=0.7)

    assert len(train_df) + len(test_df) == len(df)
    # Every train Time must be <= every test Time (strict time ordering, no shuffling)
    assert train_df["Time"].max() <= test_df["Time"].min()


def test_baseline_detector_runs(small_df):
    df = add_pseudo_entity_ids(small_df, n_clusters=10)
    df = add_velocity_features(df)
    baseline = EwmaCusumBaseline()
    scores = baseline.score(df)
    assert len(scores) == len(df)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_get_feature_columns_matches_dataframe(small_df):
    feat_df = build_feature_matrix(small_df)
    cols = get_feature_columns(feat_df)
    missing = set(cols) - set(feat_df.columns)
    assert not missing, f"Missing expected feature columns: {missing}"
