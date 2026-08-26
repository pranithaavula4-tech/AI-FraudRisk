"""
app.py
------
Minimal Streamlit demo for the Fraud-Spike Detector.

Lets the user:
  - Pick a transaction from the held-out test set
  - See its fraud-spike score from the trained model
  - See the velocity/burst features that drove that score
  - See the current cost-optimal decision threshold and whether this
    transaction would be flagged

DEFENSE-ONLY: this app only scores and displays existing transactions.
It does not generate synthetic transactions or fraud patterns, and does
not expose anything that could help someone evade the detector.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.append(os.path.join(ROOT_DIR, "src"))

st.set_page_config(page_title="Fraud-Spike Detector", layout="wide")

REQUIRED_ARTIFACTS = [
    os.path.join(MODELS_DIR, "xgb_fraud_spike_model.joblib"),
    os.path.join(MODELS_DIR, "feature_columns.joblib"),
    os.path.join(MODELS_DIR, "test_set_with_features.parquet"),
    os.path.join(MODELS_DIR, "train_meta.json"),
    os.path.join(RESULTS_DIR, "metrics_report.json"),
]


def ensure_artifacts():
    """
    Deployment platforms (Streamlit Community Cloud, AWS App Runner, etc.)
    get a fresh clone of the repo with no models/results/ contents -- we
    deliberately don't commit those large generated files to git. On first
    load, if they're missing, this runs the training + evaluation pipeline
    once so the app becomes fully self-contained. Subsequent reruns within
    the same running instance skip this (checked via file existence).
    """
    if all(os.path.exists(p) for p in REQUIRED_ARTIFACTS):
        return
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with st.spinner(
        "First-time setup: training the model and computing the cost-optimal "
        "threshold (only happens once per deployment, ~30-60s)..."
    ):
        import train as train_mod
        import evaluate as evaluate_mod
        train_mod.main()
        evaluate_mod.main()


@st.cache_resource
def load_artifacts():
    ensure_artifacts()
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_fraud_spike_model.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    test_df = pd.read_parquet(os.path.join(MODELS_DIR, "test_set_with_features.parquet"))
    with open(os.path.join(MODELS_DIR, "train_meta.json")) as f:
        meta = json.load(f)
    report = None
    report_path = os.path.join(RESULTS_DIR, "metrics_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    return model, feature_cols, test_df, meta, report


def main():
    st.title("🚨 Fraud-Spike Detector")
    st.caption(
        "Defense-only demo: scores and flags existing transactions from the held-out "
        "test set. Does not generate synthetic fraud or evasion techniques."
    )

    model, feature_cols, test_df, meta, report = load_artifacts()

    if report is None:
        st.warning("No results/metrics_report.json found -- run `python src/evaluate.py` first "
                    "to compute the cost-optimal threshold. Using default threshold 0.5 for now.")
        cost_optimal_threshold = 0.5
    else:
        cost_optimal_threshold = report["cost_model"]["cost_optimal_threshold"]

    st.sidebar.header("About this run")
    st.sidebar.write(f"**Data source:** {meta['data_source']}")
    st.sidebar.write(f"**Test set size:** {meta['n_test']:,} transactions")
    st.sidebar.write(f"**Test set frauds:** {meta['n_test_frauds']}")
    if report:
        st.sidebar.write(f"**PR-AUC:** {report['ml_model']['pr_auc']:.4f}")
    st.sidebar.write(f"**Cost-optimal threshold:** {cost_optimal_threshold:.4f}")

    st.subheader("1. Pick a transaction")
    col_a, col_b = st.columns(2)
    with col_a:
        filter_choice = st.radio(
            "Filter by:",
            ["Any transaction", "Known frauds only", "Random sample"],
            horizontal=True,
        )
    with col_b:
        n_show = st.slider("Number of rows to browse", 5, 50, 15)

    if filter_choice == "Known frauds only":
        subset = test_df[test_df["Class"] == 1].head(n_show)
    elif filter_choice == "Random sample":
        subset = test_df.sample(min(n_show, len(test_df)), random_state=None)
    else:
        subset = test_df.head(n_show)

    display_cols = ["Time", "Amount", "Class", "entity_id",
                     "entity_1min_count", "global_1min_count"]
    st.dataframe(subset[display_cols], use_container_width=True)

    row_index = st.selectbox("Select a row index to inspect:", subset.index.tolist())
    row = test_df.loc[row_index]

    st.subheader("2. Score this transaction")
    X_row = pd.DataFrame([row[feature_cols]])
    score = float(model.predict_proba(X_row)[:, 1][0])
    flagged = score >= cost_optimal_threshold

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud-spike score", f"{score:.4f}")
    m2.metric("Decision threshold", f"{cost_optimal_threshold:.4f}")
    m3.metric("Flagged?", "🚩 YES" if flagged else "✅ no")
    m4.metric("Actual label", "FRAUD" if row["Class"] == 1 else "legit")

    st.subheader("3. Velocity / burst features driving this score")
    velocity_cols = [c for c in feature_cols if c.startswith("entity_") or c.startswith("global_")]
    velocity_view = row[velocity_cols].rename("value").to_frame()
    st.dataframe(velocity_view, use_container_width=True)

    st.caption(
        "entity_* features use a real entity ID if your dataset provides one (see config.py "
        "ENTITY_ID_COL), otherwise a synthetic proxy clustered on the static features -- "
        "see README for details. global_* features are the real system-wide transaction "
        "rate and are never a proxy."
    )

    if report:
        st.subheader("4. Cost model summary (from evaluate.py)")
        cm = report["cost_model"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost / false positive", f"Rs {cm['cost_false_positive_rs']:.0f}")
        c2.metric("Cost-optimal total cost", f"Rs {cm['cost_optimal_total_cost_rs']:,.0f}")
        c3.metric("Cost at threshold=0.5", f"Rs {cm['cost_at_threshold_0.5_rs']:,.0f}")


if __name__ == "__main__":
    main()
