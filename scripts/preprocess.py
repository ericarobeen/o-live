# preprocess.py

# -----------------------------
# Backup: Install Manually
# -----------------------------
#import subprocess
#import sys

#subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
import subprocess
import sys

required_packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "pyarrow",
    "s3fs",
    "matplotlib",
    "joblib",
]

# Install packages if missing
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import os
import re
import argparse
from pathlib import Path
from datetime import datetime
import s3fs
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# Output directory for processed data or testing in Studio
if os.path.exists("/opt/ml/processing/output"):
    OUT_DIR = "/opt/ml/processing/output"
else:
    OUT_DIR = os.path.expanduser("~/SageMaker/olive-forecasting/output")


# -----------------------------
# Helper: compute cost pressure
# -----------------------------
def compute_cost_pressure(df):
    """Compute cost_pressure as weighted z-score of key drivers."""
    def zscore(s):
        return (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) != 0 else 0

    drivers = ["ocean_proxy", "diesel_usd_per_gal", "ppi_glass", "ppi_plastic_bottles", "ppi_steel"]
    weights = {
        "ocean_proxy": 0.4,
        "diesel_usd_per_gal": 0.3,
        "ppi_glass": 0.1,
        "ppi_plastic_bottles": 0.1,
        "ppi_steel": 0.1
    }

    valid_cols = []
    for col in drivers:
        if col in df.columns and not df[col].isna().all() and not (df[col] == 0).all():
            df[f"z_{col}"] = zscore(df[col])
            valid_cols.append(f"z_{col}")

    if valid_cols:
        total_weight = sum(weights[col] for col in drivers if f"z_{col}" in df.columns)
        df["cost_pressure"] = sum(df[f"z_{col}"] * weights[col] / total_weight 
                                   for col in drivers if f"z_{col}" in df.columns)
    else:
        df["cost_pressure"] = 0

    return df

# -----------------------------
# Helper: list snapshots in S3
# -----------------------------
def list_snapshot_paths(bucket, prefix="features/weekly_panel/"):
    fs = s3fs.S3FileSystem()
    glob = f"{bucket}/{prefix}snapshot_date=*/features.parquet"
    return fs.glob(glob)

def extract_snapshot_date(path):
    m = re.search(r"snapshot_date=(\d{4}-\d{2}-\d{2})", path)
    return m.group(1) if m else None

def load_parquet_file(s3_path):
    fs = s3fs.S3FileSystem()
    with fs.open(s3_path, "rb") as f:
        table = pq.read_table(f)
    return table.to_pandas()

# -----------------------------
# Helper: create features
# -----------------------------
def create_features(df, price_col="price_usd_per_l"):
    df = df.sort_index()
    s = df[price_col]
    df = df.copy()
    df["lag1week"] = s.shift(1)
    df["lag2week"] = s.shift(2)
    df["rolling3"] = s.rolling(3, min_periods=1).mean()
    df["rolling10"] = s.rolling(10, min_periods=1).mean()
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["sin_week"] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
    return df

# -----------------------------
# Helper: aggregate & feature engineering
# -----------------------------
def aggregate_and_feature(df_raw):
    # Ensure price column exists
    if "price_usd_per_l" not in df_raw.columns:
        if {"price_eur_per_l","usd_per_eur"}.issubset(df_raw.columns):
            df_raw["price_usd_per_l"] = df_raw["price_eur_per_l"] * df_raw["usd_per_eur"]
        else:
            raise RuntimeError("No price column found")

    # Ensure week_start is datetime index
    if "week_start" in df_raw.columns:
        df_raw["week_start"] = pd.to_datetime(df_raw["week_start"])
        df_raw = df_raw.set_index("week_start")
    else:
        df_raw.index = pd.to_datetime(df_raw.index)

    # Ensure country/market/grade columns exist
    for col in ["country","market","grade"]:
        if col not in df_raw.columns:
            df_raw[col] = "ALL"

    # Weekly aggregation per group
    weekly = (
        df_raw
        .groupby(["country","market","grade", pd.Grouper(freq="W")])
        .agg({"price_usd_per_l":"mean","cost_pressure":"mean"})
        .reset_index()
    )

    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly = weekly.set_index("week_start").sort_index()

    # Feature engineering per group
    frames = []
    for name, grp in weekly.groupby(["country","market","grade"]):
        grp_local = grp[["price_usd_per_l","cost_pressure"]].copy()
        grp_local = create_features(grp_local)
        grp_local["country"], grp_local["market"], grp_local["grade"] = name
        frames.append(grp_local.reset_index())

    out = pd.concat(frames, ignore_index=True)
    return out

# -----------------------------
# Main execution
# -----------------------------
def main(args):
    bucket = args.bucket.rstrip("/")
    matches = list_snapshot_paths(bucket)
    if not matches:
        raise SystemExit("No snapshots found")

    # Use latest snapshot
    snapshots = sorted(
        [(extract_snapshot_date(p), p) for p in matches if extract_snapshot_date(p)],
        key=lambda x: x[0]
    )
    latest_date, latest_path = snapshots[-1]
    print("Loading snapshot:", latest_date, latest_path)

    df = load_parquet_file(latest_path)

    # Compute derived features before aggregation
    df = compute_cost_pressure(df)

    processed = aggregate_and_feature(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "processed.parquet")
    processed.to_parquet(out_path, index=False)
    print("Wrote processed data:", out_path)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    main(args)
