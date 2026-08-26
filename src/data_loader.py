"""
data_loader.py
---------------
Loads a fraud/risk transaction dataset and standardizes it into the
canonical column set the rest of the pipeline expects: Time, Amount,
Class (if present), plus all other columns untouched.

SCHEMA FLEXIBILITY
This loader is schema-driven via config.py -- it does NOT hard-code the
ULB dataset's column names. To point this pipeline at a different
fraud/risk dataset:
  1. Edit config.py: set DATA_PATH to your CSV, and set TIME_COL,
     AMOUNT_COL, LABEL_COL, ENTITY_ID_COL to your dataset's actual
     column names (see config.py for full docs on each setting).
  2. That's it -- data_loader.py renames your columns to the canonical
     names internally, and feature_engineering.py / train.py /
     evaluate.py / detect.py all work unchanged.

If data/creditcard.csv (or your configured DATA_PATH) is not present,
this falls back to a locally generated SYNTHETIC dataset that mimics a
ULB-style schema, so the pipeline can still be exercised end-to-end with
zero setup. This fallback is clearly logged and labeled in all
downstream reports.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

N_SYNTHETIC_ROWS = 120_000       # smaller than the real 284,807 for fast iteration
FRAUD_RATE = 0.00173             # matches real dataset's ~0.173% fraud rate
RANDOM_SEED = 42

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", config.DATA_PATH)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the user-configured TIME_COL / AMOUNT_COL / LABEL_COL /
    ENTITY_ID_COL to the canonical names (Time, Amount, Class, entity_id)
    the rest of the pipeline expects, and converts a datetime TIME_COL to
    elapsed seconds if configured to do so.
    """
    df = df.copy()
    rename_map = {}

    if config.TIME_COL not in df.columns:
        raise ValueError(
            f"config.TIME_COL='{config.TIME_COL}' not found in dataset columns: {list(df.columns)}"
        )
    rename_map[config.TIME_COL] = "Time"

    if config.AMOUNT_COL not in df.columns:
        raise ValueError(
            f"config.AMOUNT_COL='{config.AMOUNT_COL}' not found in dataset columns: {list(df.columns)}"
        )
    rename_map[config.AMOUNT_COL] = "Amount"

    if config.LABEL_COL is not None and config.LABEL_COL in df.columns:
        rename_map[config.LABEL_COL] = "Class"

    if config.ENTITY_ID_COL is not None:
        if config.ENTITY_ID_COL not in df.columns:
            raise ValueError(
                f"config.ENTITY_ID_COL='{config.ENTITY_ID_COL}' not found in dataset columns: "
                f"{list(df.columns)}. Set ENTITY_ID_COL = None in config.py if you don't have one."
            )
        rename_map[config.ENTITY_ID_COL] = "real_entity_id"

    df = df.rename(columns=rename_map)

    if config.TIME_COL_IS_DATETIME:
        df["Time"] = pd.to_datetime(df["Time"])
        df["Time"] = (df["Time"] - df["Time"].min()).dt.total_seconds()

    if "Class" not in df.columns:
        # Unlabeled data is fine for real-time scoring (detect.py), but
        # train.py / evaluate.py require labels -- they will raise a
        # clear error if Class is missing when they need it.
        pass

    return df


def load_transactions(path: str = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with canonical columns: Time, Amount, and
    (if available) Class, real_entity_id, plus all original feature
    columns untouched.
    """
    if os.path.exists(path):
        if verbose:
            print(f"Loading dataset from {path}")
        df = pd.read_csv(path)
        df = _standardize_columns(df)
        df.attrs["source"] = "user_dataset" if path != os.path.join(
            os.path.dirname(__file__), "..", "data", "creditcard.csv"
        ) else "bundled_dataset"
        return df

    if verbose:
        print(
            f"WARNING: {path} not found.\n"
            "Falling back to a SYNTHETIC dataset generated to mimic the schema "
            "and class imbalance of a ULB-style credit card fraud dataset.\n"
            "Update config.DATA_PATH to point at your own dataset to use it instead."
        )
    df = generate_synthetic_fallback()
    df.attrs["source"] = "synthetic_fallback"
    return df


def generate_synthetic_fallback(
    n_rows: int = N_SYNTHETIC_ROWS,
    fraud_rate: float = FRAUD_RATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a synthetic transaction dataset with the same column layout
    as the real ULB dataset:
      - Time: seconds elapsed, spanning ~2 days, with realistic diurnal
        volume patterns and injected short-duration "spike" bursts to
        give the burst/velocity features something real to detect.
      - V1..V28: PCA-like standard-normal features. For fraud rows, a
        subset of components is shifted to create a (mildly) separable
        signal, mirroring the fact that in the real dataset fraud is
        statistically distinguishable but not trivially so.
      - Amount: log-normal, with fraud amounts drawn from a different
        (typically smaller, "testing the card") distribution.
      - Class: 1 = fraud, 0 = legitimate.
    """
    rng = np.random.default_rng(seed)

    total_seconds = 2 * 24 * 60 * 60  # 2 days, like the real dataset (~48h)

    # --- Build a non-uniform arrival process (busier during "daytime") ---
    minute_grid = np.arange(0, total_seconds, 60)
    hour_of_day = (minute_grid // 3600) % 24
    # simple diurnal multiplier: quiet at night, busy 9am-9pm
    diurnal = 0.3 + 0.7 * np.clip(np.sin((hour_of_day - 6) / 24 * 2 * np.pi) + 0.5, 0, 1)
    base_rate_per_minute = (n_rows / (total_seconds / 60)) * diurnal
    counts_per_minute = rng.poisson(base_rate_per_minute)

    # --- Inject a handful of synthetic fraud "spike bursts" ---
    n_spikes = 6
    spike_minutes = rng.choice(len(minute_grid), size=n_spikes, replace=False)
    for m in spike_minutes:
        window = slice(max(0, m - 1), min(len(counts_per_minute), m + 2))
        counts_per_minute[window] = counts_per_minute[window] + rng.integers(15, 40)

    times = []
    for minute_idx, cnt in enumerate(counts_per_minute):
        if cnt <= 0:
            continue
        offsets = rng.uniform(0, 60, size=cnt)
        times.extend(minute_grid[minute_idx] + offsets)
    times = np.sort(np.array(times))
    times = times[:n_rows] if len(times) > n_rows else times
    n_rows_actual = len(times)

    # --- Assign fraud labels, weighted toward the spike windows ---
    is_spike_time = np.zeros(n_rows_actual, dtype=bool)
    spike_second_ranges = [(minute_grid[m] - 60, minute_grid[m] + 120) for m in spike_minutes]
    for lo, hi in spike_second_ranges:
        is_spike_time |= (times >= lo) & (times <= hi)

    base_fraud_prob = np.full(n_rows_actual, fraud_rate * 0.4)
    base_fraud_prob[is_spike_time] = fraud_rate * 25  # much higher inside spikes
    base_fraud_prob = np.clip(base_fraud_prob, 0, 0.9)
    y = rng.binomial(1, base_fraud_prob)

    # nudge total fraud count toward target rate
    target_frauds = max(20, int(n_rows_actual * fraud_rate))
    current_frauds = y.sum()
    if current_frauds < target_frauds:
        extra_idx = rng.choice(
            np.where(y == 0)[0], size=target_frauds - current_frauds, replace=False
        )
        y[extra_idx] = 1
    elif current_frauds > target_frauds:
        drop_idx = rng.choice(
            np.where(y == 1)[0], size=current_frauds - target_frauds, replace=False
        )
        y[drop_idx] = 0

    n_fraud = y.sum()

    # --- V1..V28: PCA-like features ---
    V = rng.normal(0, 1, size=(n_rows_actual, 28))
    fraud_idx = np.where(y == 1)[0]
    shifted_components = rng.choice(28, size=6, replace=False)
    for comp in shifted_components:
        V[fraud_idx, comp] += rng.normal(3.0, 1.0) * rng.choice([-1, 1])

    # --- Amount ---
    amount = rng.lognormal(mean=3.0, sigma=1.2, size=n_rows_actual)
    amount[fraud_idx] = rng.lognormal(mean=2.0, sigma=1.5, size=n_fraud)
    amount = np.round(amount, 2)

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.round(times, 3))
    df["Amount"] = amount
    df["Class"] = y

    df = df.sort_values("Time").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_transactions()
    print(df.shape)
    print(df["Class"].value_counts())
    print(f"Source: {df.attrs.get('source')}")
