"""
download_data.py
-----------------
Downloads the ULB Credit Card Fraud Detection dataset from Kaggle
(mlg-ulb/creditcardfraud) into the local data/ folder.

WHY THIS SCRIPT EXISTS
This project's feature engineering and training pipeline expect a CSV at
data/creditcard.csv with columns: Time, V1..V28, Amount, Class.

HOW TO USE
1. Install kagglehub:  pip install kagglehub
2. Set up Kaggle credentials (one of):
     a) Place your kaggle.json (from https://www.kaggle.com/settings -> Create New Token)
        at ~/.kaggle/kaggle.json
     b) Or set environment variables KAGGLE_USERNAME and KAGGLE_KEY
3. Run:  python data/download_data.py

If you don't want to use the Kaggle API, you can instead:
  - Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  - Download creditcard.csv manually
  - Place it at data/creditcard.csv

NOTE ON THIS SUBMISSION
In the hackathon sandbox used to build/test this project, outbound network
access to kaggle.com was not available, so the pipeline was validated end
-to-end against a locally generated SYNTHETIC dataset that mimics the real
dataset's schema, size order-of-magnitude, and class imbalance (see
src/data_loader.py -> generate_synthetic_fallback()). Running this script
with valid Kaggle credentials will fetch the real data and the exact same
pipeline (feature engineering, training, evaluation) will run against it
unchanged.
"""

import os
import shutil
import sys

TARGET_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")


def download_with_kagglehub():
    import kagglehub

    print("Downloading mlg-ulb/creditcardfraud via kagglehub ...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print(f"Downloaded to: {path}")

    # kagglehub returns a directory; find the CSV inside it
    csv_candidate = None
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.lower() == "creditcard.csv":
                csv_candidate = os.path.join(root, f)
                break
        if csv_candidate:
            break

    if csv_candidate is None:
        raise FileNotFoundError(
            "Could not find creditcard.csv inside the downloaded dataset directory."
        )

    shutil.copy(csv_candidate, TARGET_PATH)
    print(f"Copied dataset to {TARGET_PATH}")


if __name__ == "__main__":
    if os.path.exists(TARGET_PATH):
        print(f"{TARGET_PATH} already exists. Delete it first to re-download.")
        sys.exit(0)

    try:
        download_with_kagglehub()
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print(
            "\nPlease download manually from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place creditcard.csv at: {TARGET_PATH}\n"
        )
        sys.exit(1)
