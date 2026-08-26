"""
config.py
---------
Central place to tell the pipeline how to interpret YOUR dataset's
columns. Edit this file when you swap in a different fraud/risk dataset
-- nothing else in src/ needs to change for most datasets.

WHAT EACH SETTING MEANS
------------------------
DATA_PATH
    Path to your CSV. Defaults to data/creditcard.csv (the bundled
    dataset). Point this at your own file when you're ready.

TIME_COL
    Name of the column holding a numeric timestamp -- seconds (or any
    consistent unit) elapsed, used to sort and compute rolling windows.
    If your data has a real datetime column instead (e.g. "2024-01-05
    14:32:01"), set TIME_COL to that column's name and set
    TIME_COL_IS_DATETIME = True; the loader will convert it to elapsed
    seconds automatically.

TIME_COL_IS_DATETIME
    Set True if TIME_COL contains datetime strings/objects rather than
    plain numeric seconds.

AMOUNT_COL
    Name of the column holding the transaction's monetary value. Used
    both as a model feature and as the false-negative cost in the cost
    model (a missed fraud "costs" its own Amount).

LABEL_COL
    Name of the column holding the fraud label (1 = fraud, 0 = legit).
    If you're running on unlabeled data purely for real-time scoring
    (detect.py), this can be left as-is -- it's only required for
    train.py / evaluate.py.

ENTITY_ID_COL
    Name of a REAL entity identifier column (card ID, account ID,
    customer ID, device ID, etc.), if your dataset has one. When set,
    the pipeline uses true per-entity grouping for velocity features
    instead of the KMeans pseudo-entity proxy. Set to None if you don't
    have one (e.g. the bundled ULB-style dataset), and the pipeline
    will fall back to clustering on the numeric feature columns to
    approximate entities.

FEATURE_COLS
    List of column names to use as the model's "static" per-transaction
    features (on top of the velocity features the pipeline computes
    automatically). Set to "auto" to automatically use every numeric
    column that isn't TIME_COL, AMOUNT_COL, LABEL_COL, or ENTITY_ID_COL
    -- this is what makes the pipeline work with datasets that don't use
    the ULB dataset's "V1..V28" naming convention.

N_PSEUDO_ENTITIES
    Only used when ENTITY_ID_COL is None. Number of KMeans clusters used
    to approximate entities from the feature columns. Increase this for
    larger/more heterogeneous datasets.

VELOCITY_WINDOWS_SECONDS
    The rolling-window sizes (in seconds) used for burst/velocity
    features. Defaults to 1 minute / 5 minutes / 1 hour. Adjust to match
    the natural cadence of your data -- e.g. for daily-granularity risk
    data you might use windows in days converted to seconds.

COST_FALSE_POSITIVE
    Rs (or your currency) cost of a false positive -- i.e. the ops cost
    of a human reviewing a transaction that turns out to be legitimate.
"""

DATA_PATH = "data/creditcard.csv"

TIME_COL = "Time"
TIME_COL_IS_DATETIME = False

AMOUNT_COL = "Amount"
LABEL_COL = "Class"
ENTITY_ID_COL = None  # e.g. "card_id", "account_id", "customer_id" -- set this if your data has one

FEATURE_COLS = "auto"  # or an explicit list, e.g. ["merchant_category", "device_score", "V1", "V2"]

N_PSEUDO_ENTITIES = 40

VELOCITY_WINDOWS_SECONDS = {
    "1min": 60,
    "5min": 300,
    "1hr": 3600,
}

COST_FALSE_POSITIVE = 50.0
