# Running Fraud-Spike Detector Locally — Setup Guide

This guide walks you through running the Fraud-Spike Detector project on
your own laptop (Windows, macOS, or Linux), end to end: environment setup
→ getting the real dataset → training → evaluation → the Streamlit demo
→ tests. It also explains the core concepts behind fraud-spike detection
so the code isn't a black box.

---

## Part 1: Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10, 3.11, or 3.12 | Check with `python3 --version` (macOS/Linux) or `python --version` (Windows). XGBoost/LightGBM wheels are most reliable on these versions. |
| pip | Comes with Python. Upgrade with `python -m pip install --upgrade pip`. |
| ~500 MB free disk | For the dataset, model artifacts, and the parquet feature caches. |
| (Optional) Kaggle account | Needed only if you want to train on the **real** ULB dataset instead of the built-in synthetic fallback. |

You do **not** need a GPU — XGBoost runs fine on CPU for this dataset size.

---

## Part 2: Unpack the project

1. Unzip `fraud-spike-detector.zip` wherever you keep projects, e.g.:
   ```bash
   cd ~/projects
   unzip fraud-spike-detector.zip
   cd fraud-spike-detector
   ```
2. You should see this structure:
   ```
   fraud-spike-detector/
   ├── data/download_data.py
   ├── src/ (data_loader.py, feature_engineering.py, train.py, evaluate.py, detect.py)
   ├── app.py
   ├── models/        <- pre-populated from our test run; safe to delete and regenerate
   ├── results/       <- pre-populated from our test run; safe to delete and regenerate
   ├── tests/test_pipeline.py
   ├── requirements.txt
   └── README.md
   ```

---

## Part 3: Create an isolated environment

Using a virtual environment keeps this project's package versions from
conflicting with anything else on your machine.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run PowerShell as Administrator
once and execute `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`,
then retry.

You'll know it worked when your terminal prompt shows `(.venv)` at the start.

---

## Part 4: Install dependencies

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, scikit-learn, xgboost, lightgbm, streamlit,
matplotlib, imbalanced-learn, pyarrow, joblib, kagglehub, and pytest.

**Common install hiccups:**
- **Apple Silicon (M1/M2/M3) + xgboost/lightgbm build errors** — install
  via conda instead if pip fails: `conda install -c conda-forge xgboost lightgbm`,
  then `pip install -r requirements.txt` again (pip will skip the two
  already-installed packages).
- **`pyarrow` build failing on an old pip** — run `pip install --upgrade pip`
  first, then retry.

---

## Part 5: The dataset

**No setup needed.** This project ships with a ready-to-run dataset at
`data/creditcard.csv`, so `python src/train.py` works immediately after
Part 4. It's a synthetically generated dataset matching the real ULB
Credit Card Fraud dataset's schema and class imbalance (~0.17% fraud).

From here you have three paths:

### Path A — Just use the bundled dataset
Do nothing else. Proceed to Part 6. Good for exploring the code, the
demo, and the concepts without any external dependency.

### Path B — Swap in the real ULB (Kaggle) dataset
1. Create a free Kaggle account at https://www.kaggle.com if you don't have one.
2. Go to **https://www.kaggle.com/settings** → **API** → **Create New Token**.
   This downloads a `kaggle.json` file containing your API credentials.
3. Place it here:
   - macOS/Linux: `~/.kaggle/kaggle.json`
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
4. Run:
   ```bash
   python data/download_data.py
   ```
   This downloads the real dataset and **overwrites** `data/creditcard.csv`
   with it. You should see `Copied dataset to .../data/creditcard.csv`.

   Alternative without the API: manually download `creditcard.csv` from
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and drop it
   directly into `data/creditcard.csv`, replacing the bundled file.

### Path C — Plug in your own fraud/risk dataset (any schema)
This is the one you'll want once you're ready to move off the bundled/
Kaggle data entirely — including onto real production interaction data.

1. Put your CSV somewhere in the project, e.g. `data/my_transactions.csv`.
2. Open **`config.py`** in the project root and edit these settings to
   match your dataset's actual column names:
   ```python
   DATA_PATH = "data/my_transactions.csv"

   TIME_COL = "transaction_timestamp"   # your timestamp column
   TIME_COL_IS_DATETIME = True          # True if it's a date/time string, not raw seconds

   AMOUNT_COL = "amount"                # your monetary value column
   LABEL_COL = "is_fraud"               # your fraud label column (1 = fraud, 0 = legit)
   ENTITY_ID_COL = "account_id"         # your real card/account/customer ID column, or None

   FEATURE_COLS = "auto"                # or an explicit list of your own feature columns
   ```
3. That's it — no other file needs to change. Continue to Part 6 exactly
   as normal (`python src/train.py`, etc.). The pipeline reads your
   column names through this config, builds the same velocity/burst
   features over your actual timeline, and — if you set `ENTITY_ID_COL`
   — uses your *real* entity groupings instead of the KMeans proxy the
   bundled dataset needs (since it has no true entity ID).

If your dataset doesn't have labels (e.g. you only want real-time scoring
via `detect.py`, not training), you can still leave `LABEL_COL` pointing
at a column name that doesn't exist — `train.py`/`evaluate.py` will error
clearly if you try to use them without labels, but `detect.py` alone
doesn't need labels to score a transaction against an already-trained model.

---

## Part 6: Run the pipeline

From the project root, with your virtual environment active:

```bash
# 1. Train the model (time-based split, XGBoost + EWMA/CUSUM baseline)
python src/train.py

# 2. Evaluate: PR-AUC, thresholds, confusion matrices, cost model
python src/evaluate.py

# 3. (Optional) smoke-test real-time-style single-transaction scoring
python src/detect.py
```

`train.py` will print how many rows/frauds ended up in the train vs. test
split, then save the model and both train/test feature sets to `models/`.
`evaluate.py` prints a metrics table plus the cost-optimal threshold, and
writes `results/metrics_report.json`, `results/pr_curve.png`,
`results/cost_vs_threshold.png`, and `results/cost_sweep.csv`.

If you re-run `train.py` after downloading the real dataset, just delete
the old `models/` and `results/` contents first (or they'll simply be
overwritten) so you don't mix synthetic and real artifacts:
```bash
rm -rf models/* results/*      # macOS/Linux
Remove-Item models\*, results\* -Recurse -Force   # Windows PowerShell
```

---

## Part 7: Run the Streamlit demo

```bash
streamlit run app.py
```

This opens a browser tab (usually `http://localhost:8501`). You can:
- Browse test-set transactions, filter to known frauds or a random sample
- Pick one and see its fraud-spike score against the cost-optimal threshold
- Inspect the exact velocity/burst features that drove the score

To stop it, go back to the terminal and press `Ctrl+C`.

If port 8501 is already in use:
```bash
streamlit run app.py --server.port 8502
```

---

## Part 8: Run the tests

```bash
pytest tests/test_pipeline.py -v
```

These are fast (a few seconds) sanity checks — they don't need the real
dataset or a trained model; they generate a small synthetic sample
in-memory and check things like: velocity features are non-negative,
shorter windows never contain more transactions than longer ones, the
EWMA z-score is causal (never uses future data), and the time-based
split has zero overlap.

---

## Troubleshooting quick-reference

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | Virtual env not activated, or `pip install -r requirements.txt` didn't complete — re-run it and check for errors. |
| `data/download_data.py` fails with a 401/403 | `kaggle.json` missing or malformed — redo Part 5 Option A step 2–3. |
| `evaluate.py` errors that model/baseline file not found | You ran `evaluate.py` before `train.py` — run `train.py` first. |
| Streamlit shows a blank page | Check the terminal for a stack trace; most often it's `models/` or `results/` missing a file because `train.py`/`evaluate.py` weren't run yet. |
| Very slow `train.py` on a big real dataset | Expected — the real dataset has ~285K rows; the rolling-window feature computation is the slowest step. Should still finish in well under a few minutes on a normal laptop. |

---

## Part 9: Conceptual notes — how fraud-spike detection works here

### The core idea: fraud often arrives in bursts, not one-offs
A single suspicious transaction is one signal. But fraud frequently shows
up as a **pattern over time**: a stolen card gets tested with several
small charges in quick succession, or a compromised merchant terminal
suddenly processes an abnormal volume of transactions. Looking only at a
transaction's own static features (amount, category, etc.) misses this
*shape*. This project adds **velocity features** — rolling counts and
sums of recent activity — so the model can see "this account/system is
suddenly doing something it doesn't normally do," not just "this one
transaction looks odd."

### Two kinds of velocity signal
- **Per-entity velocity** — how much activity has *this specific*
  card/account had in the last 1 min / 5 min / 1 hour? Useful for
  catching card testing or account takeover.
- **System-wide (global) velocity** — how much activity has happened
  *across the entire system* in the same windows? Useful for catching
  attacks that don't concentrate on one entity — e.g. a breached
  merchant, a bot attack hitting many different cards at once, or
  infrastructure-level anomalies that per-entity features would each
  individually treat as "normal."

### EWMA z-score: "unusual for whom?"
A raw count (e.g. "12 transactions in the last minute") isn't inherently
meaningful — 12 might be totally normal for a busy retail terminal and
wildly abnormal for a dormant personal card. An **EWMA (exponentially
weighted moving average) z-score** solves this by comparing the current
value to *that same entity's own recent history*: it tracks a running
mean and standard deviation that puts more weight on recent data, then
asks "how many standard deviations away from its own normal is this
right now?" This adapts automatically to each entity's (or the whole
system's) baseline instead of using one fixed threshold for everyone.

Critically, this has to be computed **causally** — using only data up to
and including the current point in time, never future data. Otherwise
you'd be leaking information from the future into a "real-time" signal,
which would look great in testing and fail in production. This project's
test suite explicitly checks for this (`test_ewma_zscore_is_causal`).

### CUSUM: catching slow drift, not just sharp spikes
An EWMA z-score is good at catching a single sharp jump. A **CUSUM
(cumulative sum) statistic** is complementary: it accumulates small
positive deviations over time, so it can catch a *slow, sustained* drift
upward that never produces one single dramatic spike but is still
abnormal in aggregate. The statistical baseline in `train.py` combines
both signals as a simple, fully interpretable, zero-training-cost point
of comparison against the ML model.

### Why the train/test split is by time, not random
Fraud patterns evolve, and so do the velocity features themselves — a
feature for transaction *N* depends on transactions before it. A random
split would let some "future" information leak into training in subtle
ways and would give an unrealistically rosy performance estimate. It's
also just not how the model will actually be used: in production you
can only ever train on the past and score the future. Training on the
earliest 70% by time and testing on the strictly later 30% mirrors real
deployment and gives an honest estimate of how the model would have
performed if it had been running live.

### Why PR-AUC instead of ROC-AUC
Fraud is extremely rare (often well under 1% of transactions). With that
much class imbalance, **ROC-AUC can stay high (e.g. 0.95+) even for a
fairly weak model**, because it's dominated by the huge number of
true negatives that are trivially easy to get right. **PR-AUC
(precision-recall AUC)** ignores true negatives and focuses on how well
the model does at actually finding the rare positive class without
drowning it in false alarms — which is what actually matters
operationally: "of the transactions I flag, how many are real fraud?"
and "of the real frauds, how many did I catch?"

### Why a cost model instead of picking a threshold by F1
F1-score treats every false positive and every false negative as equally
bad. In reality they aren't: a false positive costs a fixed, small
amount of ops time to manually review; a false negative costs the *actual
amount of money lost* to that specific fraud, which varies transaction
by transaction. This project makes that trade-off explicit:
`total_cost = (false positives × cost_per_review) + (sum of Amount over missed frauds)`,
swept across every possible decision threshold. The threshold that
minimizes this total cost — not 0.5, not the F1-optimal point — is the
one you should actually deploy, because it's the one that minimizes real
financial + operational cost given your specific cost assumptions. If
your review cost or your typical fraud size changes, just change
`COST_FALSE_POSITIVE` in `evaluate.py` and re-run — the optimal
threshold will shift accordingly.

### The pseudo-entity limitation, and why it matters
The ULB dataset was anonymized down to `Time`, 28 PCA components, `Amount`,
and `Class` — there's no real card or account ID. To still demonstrate
*how* per-entity velocity tracking would work, this project clusters
transactions on their PCA profile (`V1`...`V28`) with KMeans and treats
same-cluster transactions as a proxy "entity." This is a reasonable way
to exercise the mechanism, but a PCA-similarity cluster is **not** the
same thing as "this is genuinely the same card." In a real deployment
with a true entity ID column, you'd swap `add_pseudo_entity_ids()` for a
direct `groupby(real_entity_id)` and every downstream computation (rolling
windows, EWMA z-scores, the cost model) works unchanged — the entity
substitution is isolated to that one function by design.

### Defense-only, by design
Every piece of this system — feature engineering, model, baseline,
real-time scorer, demo — only ever **scores or flags existing
transactions**. Nothing in the codebase generates synthetic fraud
patterns for offensive use, builds adversarial examples against other
fraud models, or documents any technique for evading detection. That's
intentional and should stay that way if you extend this project.

### Why the schema is config-driven, not hard-coded
Real fraud/risk datasets almost never share the exact column names,
timestamp format, or entity-ID conventions of any one public dataset.
Hard-coding `V1..V28`, `Time`, `Amount` into every module would mean
rewriting feature engineering, training, and evaluation code every time
you pointed the project at new data. Instead, every module reads its
column names and window sizes from `config.py`, and `data_loader.py`
does the one translation step (your columns → canonical internal names)
at the very start of the pipeline. This means the exact same rolling-
window, EWMA, entity-grouping, and cost-optimization logic runs
unchanged whether you're using the bundled dataset, the real Kaggle
data, or your own production transaction export — you're only ever
editing configuration, not algorithm code.
