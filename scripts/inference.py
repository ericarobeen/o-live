import os
import glob
import joblib
import json
import pandas as pd
import numpy as np
import io
import boto3
import pyarrow.parquet as pq


# -------------------------
# 1. Load model from SageMaker model package dynamically
# -------------------------
def model_fn(model_dir):
    """
    SageMaker loads model artifacts from the Model Package into model_dir.
    We'll dynamically select models by name if multiple exist.
    """
    # Look for all .pkl artifacts in the model_dir
    all_models = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not all_models:
        raise RuntimeError(f"No model artifacts found in {model_dir}")
    # Return dict mapping model_name -> full path
    model_dict = {}
    for path in all_models:
        name = os.path.basename(path).replace(".pkl", "")
        model_dict[name] = joblib.load(path)
    return model_dict


# -------------------------
# 2. Input parser
# -------------------------
def input_fn(serialized_input_data, content_type):
    """
    Parse the incoming JSON payload into a Python dict that predict_fn expects.

    Expected JSON structure (single object):
    {
        "data_s3_path": "s3://bucket/path/processed.parquet",   # required
        "model_name": "global_ridge",                           # optional, default "global_ridge"
        "steps": 12,                                            # optional, default 12
        "country": "EL",                                        # optional, filter by country code
        "grade": "EVOO"                                         # optional, filter by grade code
    }
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    # serialized_input_data may be bytes or str depending on environment
    try:
        if isinstance(serialized_input_data, (bytes, bytearray)):
            payload_str = serialized_input_data.decode("utf-8")
        else:
            payload_str = serialized_input_data

        data = json.loads(payload_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON input: {e}")

    # We expect a single JSON object, not a list
    if not isinstance(data, dict):
        raise ValueError(
            "Expected a JSON object with keys like "
            '{"data_s3_path": "...", "model_name": "global_ridge", "steps": 12}'
        )

    # Validate required keys
    if "data_s3_path" not in data:
        raise ValueError("Missing required field 'data_s3_path' in request payload.")

    # Apply defaults if not provided
    if "model_name" not in data:
        data["model_name"] = "global_ridge"
    if "steps" not in data:
        data["steps"] = 12

    # country and grade are optional; just pass them through if present
    return data


# -------------------------
# 3. Feature engineering
# -------------------------
def _create_features(df):
    s = df["price_usd_per_l"]
    df["lag1week"] = s.shift(1)
    df["lag2week"] = s.shift(2)
    df["rolling3"] = s.rolling(3, min_periods=1).mean()
    df["rolling10"] = s.rolling(10, min_periods=1).mean()
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["sin_week"] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
    if "cost_pressure" not in df.columns:
        df["cost_pressure"] = 0.0
    return df


# -------------------------
# 4. Forecast helper
# -------------------------
def _forecast_horizon(model, hist_df, steps=12):
    temp = hist_df.copy()

    temp = temp.sort_values("week_start")
    temp = temp.drop_duplicates(subset=["week_start"], keep="last")

    temp = temp.set_index("week_start")
    temp = temp[~temp.index.duplicated(keep="last")]
    temp = temp.sort_index()

    current_index = temp.index.max()
    preds = []

    for _ in range(steps):
        next_week = current_index + pd.Timedelta(weeks=1)

        temp = temp[~temp.index.duplicated(keep="last")]
        temp = temp.sort_index()

        base_cost_pressure = temp["cost_pressure"].iloc[-1]

        new_row = pd.DataFrame(
            {
                "price_usd_per_l": [np.nan],
                "cost_pressure": [base_cost_pressure],
            },
            index=[next_week],
        )

        temp = pd.concat([temp, new_row])
        temp = temp[~temp.index.duplicated(keep="last")]
        temp = temp.sort_index()

        temp = _create_features(temp)
        temp = temp[~temp.index.duplicated(keep="last")]
        temp = temp.sort_index()

        # FEATURE_COLS is assumed to be defined elsewhere in this module
        X_next = temp.loc[[next_week], FEATURE_COLS]
        y_hat = float(model.predict(X_next)[0])

        preds.append(
            {
                "week_start": next_week.strftime("%Y-%m-%d"),
                "forecast": y_hat,
            }
        )

        temp.loc[next_week, "price_usd_per_l"] = y_hat
        current_index = next_week

    return preds


# -------------------------
# 5. Prediction function
# -------------------------
def predict_fn(input_data, model_dict):
    """
    Expects input_data dict:
      - data_s3_path: str (e.g. "s3://olive-datalake-fall2025/mlops/preprocess/processed.parquet")
      - model_name: str
      - steps: int
      - country: optional str, e.g. "EL"
      - grade: optional str, e.g. "EVOO"
    """
    # 1. Download and read historical data from S3
    s3_path = input_data["data_s3_path"]
    if not s3_path.startswith("s3://"):
        raise ValueError(f"data_s3_path must be an s3:// URI, got {s3_path}")

    # Parse bucket and key from s3://bucket/key
    _, _, rest = s3_path.partition("s3://")
    bucket, _, key = rest.partition("/")

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    # Read the StreamingBody into a seekable buffer and load parquet with pyarrow
    buffer = io.BytesIO(obj["Body"].read())
    table = pq.read_table(buffer)
    df = table.to_pandas()

    # Ensure the column name matches what the model expects
    # (Harmless shim; not strictly needed now that _create_features uses price_usd_per_l)
    if "AVG_PRICE" not in df.columns and "price_usd_per_l" in df.columns:
        df["AVG_PRICE"] = df["price_usd_per_l"]

    # --- NEW: optional filtering by country / grade -----------------
    country = input_data.get("country")
    grade = input_data.get("grade")

    if country is not None:
        df = df[df["country"] == country]

    if grade is not None:
        df = df[df["grade"] == grade]

    if df.empty:
        raise ValueError(
            f"No data found after filtering for country={country}, grade={grade}."
        )
    # ----------------------------------------------------------------

    # 2. Choose model
    model_name = input_data.get("model_name", "global_ridge")
    if model_name not in model_dict:
        raise RuntimeError(f"Model '{model_name}' not found in loaded model artifacts.")
    model = model_dict[model_name]

    # 3. Forecast
    steps = input_data.get("steps", 12)
    predictions = _forecast_horizon(model, df, steps=steps)

    # 4. Include model used and segment info in output
    for p in predictions:
        p["model_used"] = model_name
        p["country"] = country
        p["grade"] = grade

    return predictions


# -------------------------
# 6. Output formatter
# -------------------------
def output_fn(prediction, accept):
    """Serialize outputs for the SageMaker response."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
