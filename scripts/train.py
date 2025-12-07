# train.py
import os
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

# -------------------------
# Model output directory
# -------------------------
if os.path.exists("/opt/ml/model"):
    MODEL_DIR = "/opt/ml/model"  # SageMaker training container
else:
    MODEL_DIR = os.path.expanduser("~/SageMaker/olive-forecasting/models")  # Local
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Using model output directory: {MODEL_DIR}")

MIN_ROWS_PER_MODEL = 26

# -------------------------
# Load processed data
# -------------------------
def load_processed(path):
    return pd.read_parquet(path)

# -------------------------
# Feature engineering helper
# -------------------------
def add_missing_features(df_agg):
    df_agg = df_agg.sort_index()
    if "lag1week" not in df_agg.columns:
        df_agg["lag1week"] = df_agg["price_usd_per_l"].shift(1)
    if "lag2week" not in df_agg.columns:
        df_agg["lag2week"] = df_agg["price_usd_per_l"].shift(2)
    if "rolling3" not in df_agg.columns:
        df_agg["rolling3"] = df_agg["price_usd_per_l"].rolling(3, min_periods=1).mean()
    if "rolling10" not in df_agg.columns:
        df_agg["rolling10"] = df_agg["price_usd_per_l"].rolling(10, min_periods=1).mean()
    if "month" not in df_agg.columns:
        df_agg["month"] = df_agg.index.month
    if "dayofweek" not in df_agg.columns:
        df_agg["dayofweek"] = df_agg.index.dayofweek
    if "quarter" not in df_agg.columns:
        df_agg["quarter"] = df_agg.index.quarter
    if "sin_week" not in df_agg.columns:
        df_agg["sin_week"] = np.sin(2 * np.pi * df_agg.index.isocalendar().week / 52)
    if "cost_pressure" not in df_agg.columns:
        df_agg["cost_pressure"] = 0.0
    return df_agg

# -------------------------
# Fit models
# -------------------------
def fit_and_persist(df, out_dir):
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"])
        df = df.set_index("week_start")

    models_saved = []
    alphas = np.logspace(-3, 3, 20)

    # ---------- GLOBAL MODEL ----------
    global_df = df.groupby(df.index).agg({"price_usd_per_l": "mean",
                                          "cost_pressure": "mean"}).sort_index()
    global_df = add_missing_features(global_df)
    g = global_df.dropna(subset=["price_usd_per_l"])

    if len(g) >= MIN_ROWS_PER_MODEL:
        Xg = g[["lag1week","lag2week","rolling3","rolling10",
                "month","dayofweek","quarter","sin_week","cost_pressure"]].fillna(0)
        yg = g["price_usd_per_l"]
        model = RidgeCV(alphas=alphas)
        model.fit(Xg, yg)
        print("Global model alpha:", model.alpha_)
        fname = Path(out_dir) / "global_ridge.pkl"
        joblib.dump(model, fname)
        models_saved.append(str(fname))

    # ---------- SEGMENTS ----------
    segments = [
        ("country", df.groupby("country")),
        ("grade", df.groupby("grade")),
        ("country_grade", df.groupby(["country", "grade"]))
    ]

    for seg_name, grp_iter in segments:
        for keys, grp in grp_iter:
            grp2 = grp.groupby(grp.index).agg({"price_usd_per_l":"mean",
                                               "cost_pressure":"mean"}).sort_index()
            grp2 = add_missing_features(grp2)
            grp2 = grp2.dropna(subset=["price_usd_per_l"])

            if len(grp2) < MIN_ROWS_PER_MODEL:
                print(f"Skipping {seg_name} {keys} (only {len(grp2)} rows)")
                continue

            X = grp2[["lag1week","lag2week","rolling3","rolling10",
                      "month","dayofweek","quarter","sin_week","cost_pressure"]].fillna(0)
            y = grp2["price_usd_per_l"]

            model = RidgeCV(alphas=alphas)
            model.fit(X, y)
            print(f"{seg_name} {keys} alpha:", model.alpha_)

            # Clean filename
            if seg_name == "country_grade":
                country, grade = keys
                fname = f"country_grade_{str(country).replace(' ','_')}_{str(grade).replace(' ','_')}.pkl"
            else:
                fname = f"{seg_name}_{str(keys).replace(' ','_')}.pkl"

            joblib.dump(model, Path(out_dir) / fname)
            models_saved.append(fname)

    return models_saved

# -------------------------
# Main
# -------------------------
def main():
    # Detect SageMaker input channel automatically
    if "SM_CHANNEL_PROCESSED" in os.environ:
        processed_path = os.environ["SM_CHANNEL_PROCESSED"]
        print(f"Using SageMaker input channel: {processed_path}")
    else:
        # Local fallback
        parser = argparse.ArgumentParser()
        parser.add_argument("--processed_path", required=True, help="Local parquet file path")
        args = parser.parse_args()
        processed_path = args.processed_path

    df = load_processed(processed_path)
    saved = fit_and_persist(df, MODEL_DIR)
    print("Saved models:", saved)


if __name__ == "__main__":
    main()
