"""
evaluate.py
-----------
Evaluates the trained fraud-spike model on the held-out (strictly-later,
time-based) test set.

Reports:
  - Precision, Recall, F1, PR-AUC (PR-AUC, not ROC-AUC, is the headline
    metric here -- with ~0.17% positive rate, ROC-AUC is misleadingly
    high and does not reflect operational usefulness; PR-AUC is far more
    informative for extreme class imbalance).
  - Confusion matrices at thresholds 0.3, 0.5, 0.7, and the F1-optimal
    threshold.
  - A FALSE-POSITIVE COST MODEL:
      cost_false_positive = configurable constant (default Rs 50 -- the
        ops cost of a human manually reviewing a flagged transaction
        that turns out to be legitimate)
      cost_false_negative = the actual transaction Amount (the real
        fraud loss if a fraudulent transaction is missed)
    Total expected cost is computed at every threshold in a fine sweep,
    and the threshold that minimizes total cost is recommended. This
    threshold is what should actually be used operationally -- not
    0.5, which is an arbitrary default with no connection to real
    business costs.

All metrics and plots are saved to results/.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from train import EwmaCusumBaseline  # noqa: F401  (needed so joblib can unpickle the baseline)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

COST_FALSE_POSITIVE = getattr(config, "COST_FALSE_POSITIVE", 50.0)  # configurable in config.py


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_fraud_spike_model.joblib"))
    baseline = joblib.load(os.path.join(MODELS_DIR, "ewma_cusum_baseline.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    test_df = pd.read_parquet(os.path.join(MODELS_DIR, "test_set_with_features.parquet"))
    with open(os.path.join(MODELS_DIR, "train_meta.json")) as f:
        meta = json.load(f)
    return model, baseline, feature_cols, test_df, meta


def compute_pr_metrics(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    return precisions, recalls, thresholds, pr_auc


def metrics_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"threshold": float(threshold), "precision": float(p), "recall": float(r),
            "f1": float(f1), "confusion_matrix": cm.tolist()}


def find_f1_optimal_threshold(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0,
    )
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    best_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    return best_threshold, float(f1s[best_idx])


def cost_sweep(y_true: np.ndarray, y_score: np.ndarray, amounts: np.ndarray,
                cost_fp: float = COST_FALSE_POSITIVE, n_thresholds: int = 200):
    """
    Sweeps thresholds from 0 to 1 and computes total expected operational
    cost at each:
        total_cost = (num_false_positives * cost_fp)
                   + (sum of Amount over all false negatives)

    Returns a DataFrame with columns: threshold, n_fp, n_fn, cost_fp_total,
    cost_fn_total, total_cost.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        n_fp = int(fp_mask.sum())
        n_fn = int(fn_mask.sum())
        cost_fp_total = n_fp * cost_fp
        cost_fn_total = float(amounts[fn_mask].sum())
        total_cost = cost_fp_total + cost_fn_total

        rows.append({
            "threshold": t, "n_fp": n_fp, "n_fn": n_fn,
            "cost_fp_total": cost_fp_total, "cost_fn_total": cost_fn_total,
            "total_cost": total_cost,
        })
    return pd.DataFrame(rows)


def plot_pr_curve(precisions, recalls, pr_auc, out_path):
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f"PR curve (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (XGBoost Fraud-Spike Model)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cost_curve(cost_df, best_row, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(cost_df["threshold"], cost_df["total_cost"], label="Total expected cost")
    plt.plot(cost_df["threshold"], cost_df["cost_fp_total"], "--", alpha=0.6, label="False-positive cost")
    plt.plot(cost_df["threshold"], cost_df["cost_fn_total"], "--", alpha=0.6, label="False-negative cost")
    plt.axvline(best_row["threshold"], color="red", linestyle=":", label=f"Cost-optimal t={best_row['threshold']:.3f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Cost (Rs)")
    plt.title("Cost vs. Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, baseline, feature_cols, test_df, meta = load_artifacts()

    X_test = test_df[feature_cols]
    y_test = test_df["Class"].values
    amounts = test_df["Amount"].values

    y_score_ml = model.predict_proba(X_test)[:, 1]
    y_score_baseline = baseline.score(test_df)

    # ---- Headline metrics: PR-AUC ----
    precisions, recalls, _, pr_auc_ml = compute_pr_metrics(y_test, y_score_ml)
    _, _, _, pr_auc_baseline = compute_pr_metrics(y_test, y_score_baseline)

    # ---- Metrics at fixed thresholds + F1-optimal ----
    f1_opt_threshold, f1_opt_value = find_f1_optimal_threshold(y_test, y_score_ml)
    fixed_thresholds = [0.3, 0.5, 0.7]
    threshold_results = {
        str(t): metrics_at_threshold(y_test, y_score_ml, t) for t in fixed_thresholds
    }
    threshold_results["f1_optimal"] = metrics_at_threshold(y_test, y_score_ml, f1_opt_threshold)

    # ---- Cost model sweep ----
    cost_df = cost_sweep(y_test, y_score_ml, amounts, cost_fp=COST_FALSE_POSITIVE)
    best_row = cost_df.loc[cost_df["total_cost"].idxmin()]
    best_threshold = float(best_row["threshold"])
    cost_optimal_metrics = metrics_at_threshold(y_test, y_score_ml, best_threshold)

    # ---- Baseline comparison at its own best-F1 threshold ----
    baseline_f1_threshold, baseline_f1_value = find_f1_optimal_threshold(y_test, y_score_baseline)
    baseline_metrics = metrics_at_threshold(y_test, y_score_baseline, baseline_f1_threshold)

    # ---- Save plots ----
    plot_pr_curve(precisions, recalls, pr_auc_ml, os.path.join(RESULTS_DIR, "pr_curve.png"))
    plot_cost_curve(cost_df, best_row, os.path.join(RESULTS_DIR, "cost_vs_threshold.png"))
    cost_df.to_csv(os.path.join(RESULTS_DIR, "cost_sweep.csv"), index=False)

    # ---- Assemble final report ----
    report = {
        "data_source": meta["data_source"],
        "n_test": meta["n_test"],
        "n_test_frauds": meta["n_test_frauds"],
        "ml_model": {
            "pr_auc": float(pr_auc_ml),
            "thresholds": threshold_results,
            "f1_optimal_threshold": f1_opt_threshold,
        },
        "baseline_ewma_cusum": {
            "pr_auc": float(pr_auc_baseline),
            "best_f1_threshold": baseline_f1_threshold,
            "metrics_at_best_f1_threshold": baseline_metrics,
        },
        "cost_model": {
            "cost_false_positive_rs": COST_FALSE_POSITIVE,
            "cost_false_negative_rule": "actual transaction Amount",
            "cost_optimal_threshold": best_threshold,
            "cost_optimal_metrics": cost_optimal_metrics,
            "cost_optimal_total_cost_rs": float(best_row["total_cost"]),
            "cost_optimal_n_false_positives": int(best_row["n_fp"]),
            "cost_optimal_n_false_negatives": int(best_row["n_fn"]),
            "cost_at_threshold_0.5_rs": float(
                cost_df.iloc[(cost_df["threshold"] - 0.5).abs().argmin()]["total_cost"]
            ),
        },
    }

    with open(os.path.join(RESULTS_DIR, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- Print human-readable summary ----
    print("=" * 70)
    print(f"Data source: {meta['data_source']}")
    print(f"Test set: {meta['n_test']:,} rows, {meta['n_test_frauds']} frauds")
    print("=" * 70)
    print(f"\nML MODEL (XGBoost) -- PR-AUC: {pr_auc_ml:.4f}")
    print(f"Baseline (EWMA/CUSUM) -- PR-AUC: {pr_auc_baseline:.4f}")

    print("\nMetrics at fixed + F1-optimal thresholds (ML model):")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for key, m in threshold_results.items():
        print(f"{key:>10} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")

    print(f"\nCOST MODEL (cost_fp=Rs {COST_FALSE_POSITIVE}, cost_fn=Amount):")
    print(f"  Cost-optimal threshold: {best_threshold:.4f}")
    print(f"  -> precision={cost_optimal_metrics['precision']:.3f}, "
          f"recall={cost_optimal_metrics['recall']:.3f}, "
          f"FP={best_row['n_fp']}, FN={best_row['n_fn']}")
    print(f"  -> Total expected cost at optimal threshold: Rs {best_row['total_cost']:,.2f}")
    print(f"  -> Total expected cost at naive threshold=0.5: Rs {report['cost_model']['cost_at_threshold_0.5_rs']:,.2f}")
    print(f"\nSaved full report to {RESULTS_DIR}/metrics_report.json")
    print(f"Saved plots to {RESULTS_DIR}/pr_curve.png and {RESULTS_DIR}/cost_vs_threshold.png")


if __name__ == "__main__":
    main()
