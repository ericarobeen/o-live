import os
import io
import json
from datetime import datetime, timedelta
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="O-live: Olive Oil Cost & Risk Explorer",
    layout="wide",
)

# --------------------------------------------------
# SageMaker endpoint / data config
# --------------------------------------------------
REGION = "us-east-1"
ENDPOINT_NAME = os.getenv("OLIVE_ENDPOINT_NAME", "olive-test-v20")
DATA_S3_PATH = os.getenv(
    "OLIVE_DATA_S3_PATH",
    "s3://olive-datalake-fall2025/mlops/preprocess/processed.parquet",
)

COUNTRY_LABELS = {
    "EL": "Greece",
    "ES": "Spain",
    "PT": "Portugal",
    "IT": "Italy",
    "HR": "Croatia",
}

ORIGIN_MAP = {v: k for k, v in COUNTRY_LABELS.items()}


# --------------------------------------------------
# Data loading helpers
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dashboard_data(s3_uri: str) -> pd.DataFrame:
    """
    Load the main parquet file from S3 into a pandas DataFrame.
    """
    try:
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        s3 = boto3.client("s3", region_name=REGION)
        obj = s3.get_object(Bucket=bucket, Key=key)
        table = pq.read_table(io.BytesIO(obj["Body"].read()))
        df = table.to_pandas()

        # Try to coerce a date column if one exists
        for candidate in ["date", "ds", "forecast_date"]:
            if candidate in df.columns:
                df[candidate] = pd.to_datetime(df[candidate])
                df.rename(columns={candidate: "date"}, inplace=True)
                break

        return df
    except Exception as e:
        st.error(f"Failed to load data from {s3_uri}: {e}")
        return pd.DataFrame()


# --------------------------------------------------
# SageMaker inference helper
# --------------------------------------------------
def call_sagemaker_endpoint(payload: dict) -> dict:
    """
    Call the SageMaker endpoint with a JSON payload and return the parsed JSON.
    """
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        raw = response["Body"].read().decode("utf-8")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw}
    except Exception as e:
        st.error(f"Error calling SageMaker endpoint {ENDPOINT_NAME}: {e}")
        return {"error": str(e)}


# --------------------------------------------------
# UI helper functions
# --------------------------------------------------
def render_country_risk_overview(df: pd.DataFrame):
    """
    Render a simple country risk overview card section.
    Expects columns like 'origin', 'risk_score' and optionally 'latest_price'.
    This is defensive — if columns are missing, it degrades gracefully.
    """
    if df.empty:
        st.info("No data available yet for country risk overview.")
        return

    # Try to find a country column
    country_col = None
    for c in ["origin", "country", "country_code"]:
        if c in df.columns:
            country_col = c
            break

    if country_col is None:
        st.info("Country risk overview is disabled because no country column was found in the data.")
        return

    # Try to find a risk score column
    risk_col = None
    for c in ["risk_score", "risk_index", "risk"]:
        if c in df.columns:
            risk_col = c
            break

    summary = df.copy()
    if risk_col is not None:
        summary = (
            summary.groupby(country_col)[risk_col]
            .mean()
            .reset_index()
            .rename(columns={risk_col: "avg_risk"})
        )
    else:
        summary = summary[[country_col]].drop_duplicates()
        summary["avg_risk"] = np.nan

    st.subheader("Country Risk Overview")
    cols = st.columns(min(3, len(summary))) if len(summary) > 0 else []

    for i, (_, row) in enumerate(summary.iterrows()):
        col = cols[i % len(cols)] if cols else st
        label = COUNTRY_LABELS.get(row[country_col], row[country_col])
        with col:
            st.markdown(f"**{label}**")
            if not np.isnan(row["avg_risk"]):
                st.metric("Avg. Risk Score", f"{row['avg_risk']:.2f}")
            else:
                st.caption("Risk score unavailable")


def render_forecast_chart(api_response: dict):
    """
    Render a basic forecast line chart if the API response contains
    'dates' and 'prices' or similar structures.
    """
    if not api_response or "error" in api_response:
        return

    # Try a couple of common shapes
    if "forecast" in api_response and isinstance(api_response["forecast"], list):
        # Expect list of {date: ..., price: ...}
        rows = api_response["forecast"]
        if rows and isinstance(rows[0], dict):
            df = pd.DataFrame(rows)
        else:
            return
    elif "dates" in api_response and "prices" in api_response:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(api_response["dates"]),
                "price": api_response["prices"],
            }
        )
    else:
        # Nothing recognizable to plot
        return

    if "date" not in df.columns:
        # Try to coerce an index into a horizon
        df = df.reset_index().rename(columns={"index": "horizon"})
        st.line_chart(df.set_index("horizon"))
        return

    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    st.line_chart(df)


# --------------------------------------------------
# Main app
# --------------------------------------------------
def main():
    st.title("O-live: Olive Oil Cost & Risk Explorer")

    # Layout: sidebar for controls, main area for charts
    with st.sidebar:
        st.header("Planning Controls")

        # Data load just to get available countries/grades if needed
        df = load_dashboard_data(DATA_S3_PATH)

        # Country selection
        country_options = list(COUNTRY_LABELS.values())
        selected_countries_labels = st.multiselect(
            "Country of origin",
            options=country_options,
            default=country_options,
        )
        selected_country_codes = [ORIGIN_MAP[c] for c in selected_countries_labels]

        # Grade selection
        if "grade" in df.columns:
            grade_options = sorted(df["grade"].dropna().unique().tolist())
        else:
            grade_options = ["Extra Virgin", "Virgin", "Lampante"]

        selected_grade = st.selectbox("Product grade", options=grade_options)

        # Horizon
        forecast_days = st.slider(
            "Planning horizon (days ahead)",
            min_value=7,
            max_value=365,
            value=90,
            step=7,
        )

        st.caption(
            "These controls mirror real procurement levers: origin mix, grade, and planning window."
        )

        run_forecast = st.button("Run price forecast")

    # Main content layout
    df = load_dashboard_data(DATA_S3_PATH)
    col_main, col_side = st.columns([2.5, 1.5])

    with col_main:
        st.subheader("Projected Base Price")

        if run_forecast:
            with st.spinner("Calling forecasting model..."):
                payload = {
                    "countries": selected_country_codes,
                    "grade": selected_grade,
                    "days_ahead": int(forecast_days),
                }
                api_response = call_sagemaker_endpoint(payload)

            if "error" in api_response:
                st.error("The model returned an error. Check logs for details.")
                st.json(api_response)
            else:
                render_forecast_chart(api_response)
                with st.expander("Raw model response"):
                    st.json(api_response)
        else:
            st.info("Adjust the controls on the left and click **Run price forecast** to see projections.")

        st.markdown("---")
        st.subheader("Historical Context (from dashboard data)")

        if not df.empty and "date" in df.columns:
            # Optional historical plot – assumes a price column
            price_col = None
            for c in ["price", "base_price", "price_eur_per_kg", "y"]:
                if c in df.columns:
                    price_col = c
                    break

            if price_col is not None:
                hist_df = df.copy()
                hist_df = hist_df.sort_values("date")
                hist_df.set_index("date", inplace=True)
                st.line_chart(hist_df[[price_col]])
            else:
                st.caption("No obvious price column found to plot historical context.")
        else:
            st.caption("Historical plots will appear here once a date column is available in the data.")

    with col_side:
        render_country_risk_overview(df)


if __name__ == "__main__":
    main()
