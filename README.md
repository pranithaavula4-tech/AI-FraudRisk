# Fraud-Spike Detector

A **defense-only** fraud detection tool that scores and flags anomalous
bursts ("spikes") of transaction activity, both per-entity (card/account)
and system-wide, built on the schema of the ULB Credit Card Fraud
Detection dataset (Kaggle: `mlg-ulb/creditcardfraud`).

> **This project only detects and scores transactions.** It does not
> generate synthetic fraud patterns, does not build adversarial examples
> against other fraud models, and does not implement or document any
> technique for evading fraud detection. See [Defense-only statement](#defense-only-statement).

---

## ⚠️ Note on the bundled dataset

This project ships with a ready-to-run dataset at **`data/creditcard.csv`**
so it works immediately with **zero setup** — no Kaggle account, no
download step. It's a synthetically generated dataset built to match the
real ULB Credit Card Fraud dataset's schema (`Time, V1..V28, Amount,
Class`), size order of magnitude, and class imbalance (~0.17% fraud) —
see `src/data_loader.py::generate_synthetic_fallback()` for the exact
generation logic.

**All numbers in this README come from that bundled dataset.** Because
its injected fraud signal is cleaner and more separable than real fraud,
the metrics below (e.g. PR-AUC ≈ 0.995) are almost certainly **optimistic
relative to real-world data** and should not be quoted as the model's
real-world performance — they demonstrate that the pipeline and
cost-optimization logic work correctly, not that this exact PR-AUC will
hold on real transactions.

**To use the real Kaggle dataset instead:** run `python data/download_data.py`
(needs a free Kaggle account + API token — see `RUNNING_LOCALLY.md`), which
will overwrite `data/creditcard.csv` with the real data. No code changes
needed — `train.py` / `evaluate.py` pick it up automatically.

**To use your own fraud/risk dataset instead:** see
[Bring your own dataset](#bring-your-own-dataset) below — you only need to
edit `config.py`, not the pipeline code.

---

## Problem statement

Fraud doesn't just show up as one-off suspicious transactions — it often
arrives in **bursts**: a stolen card gets tested with several small
transactions in quick succession, or a compromised merchant terminal
produces an abnormal spike in transaction volume. A model trained purely
on a transaction's own static features (amount, PCA components) can
miss this "shape" of an attack. This project adds **velocity/burst
features** — how many transactions and how much money moved in the last
1 minute / 5 minutes / 1 hour, both for the transaction's entity and for
the system as a whole — on top of the standard features, and trains a
classifier to detect fraud spikes using both signals together.

### Why a time-based split (not random) matters

Fraud data is sequential, and so are the velocity features themselves:
a rolling count computed for transaction *i* depends on transactions
*before* it in time. If you split randomly, some "past" rows for a test
transaction may actually be *training* rows, and some training rows may
have been influenced by information that would only be available in the
future relative to earlier training rows. This leaks information and
gives a falsely optimistic picture of performance. It's also unrealistic:
in production you can only ever train on the past and score the future.
This project trains on the **first 70% of transactions by Time** and
tests on the **strictly later last 30%** — exactly as the model would be
evaluated in production.

---

## Feature list and rationale

| Feature group | Features | Why |
|---|---|---|
| Original | `V1`...`V28` (PCA-anonymized), `Amount` | Baseline transaction-level signal, as provided by the dataset. |
| Entity velocity | `entity_{1min,5min,1hr}_count`, `entity_{1min,5min,1hr}_sum_amount` | Trailing rolling count / total spend for the transaction's entity — the core "is this account suddenly transacting a lot?" signal. |
| Entity anomaly | `entity_1min_count_zscore` | Causal EWMA z-score of the entity's 1-minute count — flags a *statistically* unusual burst, not just a raw high count, adapting to each entity's normal baseline. |
| Global velocity | `global_{1min,5min,1hr}_count`, `global_{1min,5min,1hr}_sum_amount` | System-wide transaction rate — catches spikes that hit many entities at once (e.g. a compromised merchant or a coordinated attack), which per-entity features alone would miss. |
| Global anomaly | `global_1min_count_zscore` | Causal EWMA z-score of the system-wide 1-minute count — the global "is something unusual happening right now" signal. |

All rolling/EWMA features are computed **causally** (only using data at or
before the current transaction's timestamp) — verified in
`tests/test_pipeline.py::test_ewma_zscore_is_causal`.

### ⚠️ Entity ID: real column vs. proxy

The **bundled dataset** has no real card/account/entity identifier —
only `Time`, `V1..V28`, `Amount`, `Class` (matching the real ULB
dataset's schema). When `config.ENTITY_ID_COL` is left as `None`,
`feature_engineering.py` derives a **synthetic proxy entity ID** by
clustering transactions on their numeric feature columns
(`MiniBatchKMeans`, 40 clusters by default). This lets the pipeline
demonstrate the *mechanism* of per-entity velocity tracking, but the
resulting per-entity numbers in this README are **not equivalent to a
real per-card velocity signal** and should not be interpreted as such.

If you point `config.ENTITY_ID_COL` at a real entity column in your own
data (see [Bring your own dataset](#bring-your-own-dataset) below), the
pipeline automatically switches to grouping by that real ID instead of
clustering — giving a genuine per-card/per-account velocity feature with
no code changes required.

The **global/system-wide** features do not have this limitation either
way — they use only the real `Time` column and are a legitimate signal
regardless of entity configuration.

---

## Final metrics (held-out, time-based test set)

*Synthetic fallback data — see note above. Test set: 23,450 transactions, 35 frauds (0.15%).*

| Model | PR-AUC |
|---|---|
| **XGBoost fraud-spike model** | **0.995** |
| EWMA/CUSUM statistical baseline | 0.003 |

PR-AUC (not ROC-AUC) is the headline metric because with ~0.15% positive
rate, ROC-AUC stays misleadingly high even for a fairly weak model;
PR-AUC is much more informative about real operational usefulness.

**Precision / Recall / F1 at fixed and F1-optimal thresholds (XGBoost model):**

| Threshold | Precision | Recall | F1 | Confusion matrix (TN, FP / FN, TP) |
|---|---|---|---|---|
| 0.3 | 0.971 | 0.971 | 0.971 | 23414, 1 / 1, 34 |
| 0.5 | 1.000 | 0.971 | 0.986 | 23415, 0 / 1, 34 |
| 0.7 | 1.000 | 0.971 | 0.986 | 23415, 0 / 1, 34 |
| F1-optimal (0.999) | 1.000 | 0.971 | 0.986 | 23415, 0 / 1, 34 |

The EWMA/CUSUM baseline, evaluated at its own best-F1 threshold, reaches
only precision 0.009 / recall 0.057 / F1 0.016 — confirming the ML model
is extracting real additional signal from the combination of
velocity features and PCA components, rather than the burst features
alone doing all the work.

---

## False-positive cost model

Flagging a transaction isn't free — a human has to review it — and
missing a fraud isn't free either — the business eats the loss. We model
this explicitly instead of optimizing for F1 alone:

- **Cost of a false positive** = `₹50` (configurable constant in
  `evaluate.py::COST_FALSE_POSITIVE`) — the ops cost of a manual review
  of a transaction that turns out to be legitimate.
- **Cost of a false negative** = the transaction's actual `Amount` — the
  real fraud loss from a missed fraud.
- Total expected cost is computed at 200 thresholds swept from 0 to 1;
  see `results/cost_sweep.csv` and `results/cost_vs_threshold.png`.

**Result on this run:**

| Threshold | Total cost | False positives | False negatives |
|---|---|---|---|
| Naive default (0.5) | ₹0.89 | 0 | 1 |
| **Cost-optimal (0.457)** | **₹0.89** | **0** | **1** |

In this synthetic run the cost-optimal threshold happens to coincide
with the naive 0.5 default because the model separates classes so
cleanly that there's a wide flat region of thresholds with zero false
positives and the same single (low-`Amount`) false negative — so the
cost curve is flat across most of that range. On real (noisier) data,
we'd expect the cost-optimal threshold to diverge more visibly from 0.5,
which is exactly why this cost-sweep machinery exists: **do not assume
0.5 (or the F1-optimal threshold) is the right operating point** — always
derive it from the actual cost of a false positive vs. a false negative
in your deployment.

`detect.py`'s `FraudSpikeDetector` automatically loads this cost-optimal
threshold from `results/metrics_report.json` at inference time.

---

## Bring your own dataset

This pipeline is **not hard-coded to the ULB schema**. To point it at any
other fraud/risk dataset with a timestamp, an amount, and (for
training/evaluation) a fraud label, edit **`config.py`** — nothing else:

```python
DATA_PATH = "data/your_dataset.csv"

TIME_COL = "transaction_timestamp"   # your timestamp column
TIME_COL_IS_DATETIME = True          # True if it's a datetime string, not raw seconds

AMOUNT_COL = "amount"                # your monetary value column
LABEL_COL = "is_fraud"               # your fraud label column (1 = fraud, 0 = legit)
ENTITY_ID_COL = "account_id"         # your real card/account/customer ID column, or None

FEATURE_COLS = "auto"                # or an explicit list of your own feature column names
```

What happens automatically once you edit this:
- `data_loader.py` renames your columns to the pipeline's internal
  canonical names (`Time`, `Amount`, `Class`), converting a datetime
  timestamp to elapsed seconds if needed.
- `feature_engineering.py` builds the same velocity/burst features
  (rolling count/sum, EWMA z-score) over your data's actual timeline.
  If you set `ENTITY_ID_COL`, it groups by your **real** entity column
  instead of the KMeans pseudo-entity proxy — giving you a true per-card/
  per-account velocity signal instead of an approximation.
- `train.py` / `evaluate.py` / `detect.py` / `app.py` all work unchanged
  — they read the feature list from `feature_engineering.get_feature_columns()`,
  which adapts to whatever columns your dataset has.
- The cost model's false-positive cost is also configurable in
  `config.py` (`COST_FALSE_POSITIVE`) so you can match your own team's
  actual review cost.

You generally do not need to touch any file inside `src/` to switch
datasets — only `config.py`. This is the same mechanism you'd use to
replace the bundled synthetic data with production transaction data:
point `DATA_PATH` at your real export, map the four column settings to
your schema, and re-run `train.py` → `evaluate.py`.

---

## Project structure

```
fraud-spike-detector/
├── config.py                    # Schema mapping -- edit this to use your own dataset
├── data/
│   ├── creditcard.csv           # Bundled ready-to-run synthetic dataset (see note above)
│   └── download_data.py         # Optional: fetch the real Kaggle dataset (kagglehub)
├── src/
│   ├── data_loader.py           # Loads + standardizes any configured dataset
│   ├── feature_engineering.py   # Entity (real or proxy) + velocity/EWMA features
│   ├── train.py                 # Time-based split, XGBoost training, EWMA/CUSUM baseline
│   ├── evaluate.py              # PR-AUC, thresholds, confusion matrices, cost model
│   └── detect.py                # Real-time-style single-transaction scoring
├── app.py                       # Streamlit demo
├── models/                      # Saved model + baseline + train/test sets (generated)
├── results/                     # Metrics report, plots, cost sweep (generated)
├── tests/
│   └── test_pipeline.py         # Pipeline sanity tests (pytest)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# Works immediately with the bundled dataset -- no download needed.
python src/train.py
python src/evaluate.py
streamlit run app.py

# Optional: fetch the real ULB dataset instead (needs Kaggle credentials)
python data/download_data.py     # overwrites data/creditcard.csv
python src/train.py              # re-run with real data
python src/evaluate.py

# Optional: point at your own fraud/risk dataset -- edit config.py first,
# see "Bring your own dataset" above, then just:
python src/train.py
python src/evaluate.py

# Run tests
pytest tests/test_pipeline.py -v
```

See `RUNNING_LOCALLY.md` for a full step-by-step laptop setup guide
(virtual environment, OS-specific notes, troubleshooting) plus deeper
conceptual notes on how each part of the fraud-spike detection approach
works. See `DEPLOYMENT.md` for pushing this to GitHub and deploying it
to a public URL (Streamlit Community Cloud, AWS App Runner, or a plain
Docker container), plus notes on recording a demo video.

---

## Defense-only statement

This tool's only function is to **score and flag** transactions as
potential fraud spikes, using statistical and machine-learning signals
derived from transaction timing and amounts. It:

- Does **not** generate synthetic fraud transactions or fraud "patterns"
  intended to mimic real attacks for offensive use.
- Does **not** implement, document, or suggest any technique for evading
  fraud detection systems (this project's own or anyone else's).
- Does **not** construct adversarial examples against fraud models.
- Is intended solely to help a fraud/risk team detect and respond to
  anomalous activity faster.

## Known limitations

1. **Entity ID is a proxy unless you configure a real one** — the
   bundled dataset has no card/account ID, so entity-level velocity
   features use a KMeans-cluster proxy by default. Set
   `config.ENTITY_ID_COL` to a real column in your own data to get a
   true per-card/per-account history instead — see "Bring your own
   dataset" above. Global/system-wide features do not have this
   limitation either way.
2. **Bundled/synthetic evaluation data** — the metrics in this README
   come from the bundled synthetic dataset (or, if you've swapped in
   your own data, whatever `config.DATA_PATH` currently points at), not
   independently validated real-world fraud data. Re-run against the
   real Kaggle dataset or your own production data before trusting these
   exact numbers operationally.
3. **Small absolute fraud counts** — even the full real ULB dataset has
   only 492 frauds; any test split will have a small number of positive
   examples, so metrics (especially at the tails of the threshold sweep)
   carry real sampling noise. Treat point estimates with appropriate
   caution and prefer PR-AUC/cost-curve shape over any single number.
4. **Time column is relative, not wall-clock** — by default the dataset's
   `Time` is seconds elapsed since the first transaction, with no
   calendar date/timezone, so this cannot capture true calendar effects
   (e.g. weekday vs weekend, holidays) unless you supply a real datetime
   column via `config.TIME_COL_IS_DATETIME = True`.
5. **Single-classifier design** — this is one XGBoost model; a production
   system would likely ensemble this with rules-based checks, device/
   IP signals, and human review workflows, none of which are modeled here.
