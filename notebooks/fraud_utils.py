"""
Shared utilities for the IEEE-CIS Fraud Detection notebooks (01-04).

Centralizes things that MUST be identical across notebooks for the final
"GCN vs. baseline" comparison in 04_gcn_model.ipynb to be meaningful:

  - Data paths (DATA_DIR, PROCESSED_DIR, RESULTS_DIR)
  - Memory-efficient CSV loading (`load_raw_data`)
  - A single time-based train/validation split (`time_based_split`), used by
    both 02_baseline_model.ipynb and 03_preprocessing_gnn_features.ipynb, so
    the baseline and the GCN are evaluated on *exactly* the same holdout rows.
  - Metric / prediction persistence helpers so 04_gcn_model.ipynb can load
    02's results without re-running it.

Usage from a notebook in this same folder:

    import sys, pathlib
    sys.path.append(str(pathlib.Path.cwd()))
    from fraud_utils import DATA_DIR, load_raw_data, time_based_split, ...
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "ieee-cis-dataset"
PROCESSED_DIR = DATA_DIR / "processed"   # engineered features + graph artifacts, written by notebook 03
RESULTS_DIR = REPO_ROOT / "results"      # metrics/predictions, written by notebooks 02 & 04

RANDOM_SEED = 42
VALID_FRACTION = 0.2  # last 20% of TransactionDT (by time) held out as validation

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed python/numpy/torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to shrink memory footprint (raw CSVs are ~1.3GB)."""
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if not pd.api.types.is_numeric_dtype(col_type):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if pd.isna(c_min) or pd.isna(c_max):
            continue
        if str(col_type)[:3] == "int":
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    if verbose:
        end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        pct = 100 * (start_mem - end_mem) / start_mem if start_mem else 0
        print(f"  Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB ({pct:.0f}% reduction)")
    return df


def load_raw_data(reduce_memory: bool = True):
    """Load + merge transaction/identity CSVs for train and test. Returns (train, test)."""
    print("Loading train_transaction.csv ...")
    train_txn = pd.read_csv(DATA_DIR / "train_transaction.csv")
    print("Loading train_identity.csv ...")
    train_id = pd.read_csv(DATA_DIR / "train_identity.csv")
    print("Loading test_transaction.csv ...")
    test_txn = pd.read_csv(DATA_DIR / "test_transaction.csv")
    print("Loading test_identity.csv ...")
    test_id = pd.read_csv(DATA_DIR / "test_identity.csv")

    train = train_txn.merge(train_id, on="TransactionID", how="left")
    test = test_txn.merge(test_id, on="TransactionID", how="left")

    if reduce_memory:
        print("Reducing memory usage...")
        train = reduce_memory_usage(train)
        test = reduce_memory_usage(test)

    print(f"train: {train.shape}, test: {test.shape}, fraud rate: {train['isFraud'].mean():.4f}")
    return train, test


def time_based_split(df: pd.DataFrame, valid_fraction: float = VALID_FRACTION, time_col: str = "TransactionDT"):
    """
    Split *labeled* training data into train/valid by time, not randomly.

    IEEE-CIS's real Kaggle test set is entirely *after* the train set in time
    (TransactionDT is seconds-since-a-reference-point), so a random split
    leaks near-future information into training and overstates validation
    AUC. Sorting by TransactionDT and holding out the last `valid_fraction`
    mimics that train/test time gap.

    Returns two boolean numpy arrays (train_mask, valid_mask), aligned to
    df's existing row order (not sorted), so they can be indexed back
    directly, e.g. df.loc[train_mask, ...].
    """
    t = df[time_col].values
    cutoff = np.quantile(t, 1 - valid_fraction)
    valid_mask = t >= cutoff
    train_mask = ~valid_mask
    return train_mask, valid_mask


def save_metrics(name: str, metrics: dict) -> Path:
    path = RESULTS_DIR / f"{name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"Saved metrics -> {path}")
    return path


def load_metrics(name: str) -> dict:
    path = RESULTS_DIR / f"{name}_metrics.json"
    with open(path) as f:
        return json.load(f)


def save_predictions(name: str, y_true: np.ndarray, y_score: np.ndarray) -> Path:
    """Save validation-set (y_true, y_score) so notebook 04 can overlay ROC/PR curves."""
    path = RESULTS_DIR / f"{name}_preds.npz"
    np.savez(path, y_true=np.asarray(y_true), y_score=np.asarray(y_score))
    print(f"Saved predictions -> {path}")
    return path


def load_predictions(name: str):
    path = RESULTS_DIR / f"{name}_preds.npz"
    data = np.load(path)
    return data["y_true"], data["y_score"]
