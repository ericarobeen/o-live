import os
import json
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Determine output directory (SageMaker vs Local)
if os.path.exists("/opt/ml/processing/output"):
    OUT_DIR = "/opt/ml/processing/output"
else:
    OUT_DIR = os.path.expanduser("~/SageMaker/olive-forecasting/output")

FEATURES = [
    "lag1week", "lag2week", "rolling3", "rolling10",
    "month", "dayofweek", "quarter", "sin_week", "cost_pressure"
]

def create_features(df, price_col="price_usd_per_l"):
    """Adds lag, rolling, calendar, and Fourier features."""
    df = df.sort_index().copy()
    s = df[price_col]

    df["lag1week"] = s.shift(1)
    df["lag2week"] = s.shift(2)
    df["rolling3"] = s.rolling(3, min_periods=1).mean()
    df["rolling10"] = s.rolling(10, min_periods=1).mean()
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["sin_week"] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)

    # cost_pressure might be absent
    if "cost_pressure" not in df.columns:
        df["cost_pressure"] = 0.0

    return df


def evaluate_segment(df, model_path):
    """Runs MAE/RMSE evaluation for a model and a dataframe."""
    df = create_features(df)

    X = df[FEATURES].fillna(0)
    y_true = df["price_usd_per_l"]

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": sqrt(mean_squared_error(y_true, y_pred))
    }


def main(processed_path, model_dir):
    print("Loading processed data from:", processed_path)
    df = pd.read_parquet(processed_path)

    # Ensure datetime index
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"])
        df = df.set_index("week_start")
    else:
        df.index = pd.to_datetime(df.index)

    metrics = {}

    # ------------------------------
    # GLOBAL MODEL EVALUATION
    # ------------------------------
    global_model_path = os.path.join(model_dir, "global_ridge.pkl")
    
    if os.path.exists(global_model_path):
        print("Evaluating global model:", global_model_path)

        # Aggregate global time series
        global_df = df.groupby(df.index).agg({
            "price_usd_per_l": "mean",
            "cost_pressure": "mean"
        })

        metrics["global"] = evaluate_segment(global_df, global_model_path)
    else:
        print("WARNING: Global model file not found:", global_model_path)
        metrics["global"] = {"mae": None, "rmse": None}

    # ------------------------------
    # SEGMENT MODELS (COUNTRY / GRADE)
    # ------------------------------
    for seg_name, grp_col in [("country", "country"), ("grade", "grade")]:
        metrics[seg_name] = {}

        if grp_col not in df.columns:
            print(f"WARNING: Column '{grp_col}' missing; skipping segment {seg_name}")
            continue

        for seg_value, seg_df in df.groupby(grp_col):
            model_file = os.path.join(model_dir, f"{seg_name}_{seg_value}.pkl")

            if not os.path.exists(model_file):
                print(f"Skipping missing model: {model_file}")
                continue  # safe skip

            print(f"Evaluating {seg_name} model for {seg_value}: {model_file}")

            # Aggregate time series for the segment
            agg_df = seg_df.groupby(seg_df.index).agg({
                "price_usd_per_l": "mean",
                "cost_pressure": "mean"
            })

            metrics[seg_name][seg_value] = evaluate_segment(agg_df, model_file)

    # ------------------------------
    # WRITE METRICS
    # ------------------------------
    output_path = os.path.join(OUT_DIR, "metrics.json")
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation completed. Metrics written to:", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_path", required=True)
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()

    main(args.processed_path, args.model_dir)
