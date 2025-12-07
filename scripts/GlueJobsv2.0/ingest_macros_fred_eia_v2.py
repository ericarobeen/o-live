import argparse
from datetime import date
import os, json
import boto3, botocore
import pandas as pd
import awswrangler as wr
import re
import requests
from functools import reduce
import numpy as np

# ===============================================================
# Arg parsing
# ===============================================================
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--BUCKET")
parser.add_argument("--SNAPSHOT_DATE")
parser.add_argument("--REGION")
parser.add_argument("--API_SECRET")
args, _ = parser.parse_known_args()

bucket         = args.BUCKET
snapshot_date  = args.SNAPSHOT_DATE or date.today().isoformat()
region         = args.REGION or "us-east-1"
api_secret_arn = args.API_SECRET

if not bucket:
    raise SystemExit("Missing --BUCKET")
if not api_secret_arn:
    raise SystemExit("Missing --API_SECRET")

print(f"[INFO] Using snapshot_date={snapshot_date}")
print(f"[INFO] bucket={bucket} region={region}")
print(f"[INFO] api_secret_arn={api_secret_arn}")

# ===============================================================
# Load API keys from Secrets Manager
# ===============================================================
def get_json_secret(secret_arn, region_name):
    sm = boto3.client("secretsmanager", region_name=region_name)
    try:
        resp = sm.get_secret_value(SecretId=secret_arn)
    except botocore.exceptions.ClientError as e:
        print(f"[ERROR] Secrets Manager failure for {secret_arn}: {e}")
        raise

    val = resp.get("SecretString")
    if not val:
        raise SystemExit("SecretBinary not supported")

    obj = json.loads(val)
    if not isinstance(obj, dict):
        raise SystemExit("SecretString must be JSON")

    if "FRED_API_KEY" not in obj or "EIA_API_KEY" not in obj:
        raise SystemExit("Secret missing FRED_API_KEY or EIA_API_KEY")

    return obj

secret_obj = get_json_secret(api_secret_arn, region)
fred_key = secret_obj["FRED_API_KEY"]
eia_key  = secret_obj["EIA_API_KEY"]

print("[INFO] Successfully loaded FRED/EIA API keys")

# ===============================================================
# Snapshot Fallback Helper
# ===============================================================
# We locate the newest snapshot folder under:
#   s3://bucket/processed/macros/
#
# and return e.g.:
#   s3://bucket/processed/macros/snapshot_date=2025-11-17/
# ===============================================================

def find_latest_snapshot(bucket: str, base_prefix: str) -> str:
    """
    Return the most recent snapshot folder under:
        s3://bucket/processed/macros/
    matching pattern snapshot_date=YYYY-MM-DD/
    """

    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    snapshots = []

    print(f"[INFO] Listing snapshots under s3://{bucket}/{base_prefix}")

    for page in paginator.paginate(Bucket=bucket, Prefix=base_prefix, Delimiter="/"):
        prefixes = page.get("CommonPrefixes", [])

        for p in prefixes:
            key = p.get("Prefix", "")
            # Expect keys like processed/macros/snapshot_date=2025-11-10/
            m = re.search(r"snapshot_date=(\d{4}-\d{2}-\d{2})/", key)
            if m:
                snapshots.append(m.group(1))

    if not snapshots:
        print("[WARN] No previous snapshot found.")
        return None

    latest = sorted(snapshots)[-1]
    print(f"[INFO] Latest snapshot detected: {latest}")

    return f"{base_prefix}snapshot_date={latest}/"

def load_or_fallback(series_name: str,
                     bucket: str,
                     snapshot_date: str,
                     base_prefix: str):
    """
    Load a macro series such as 'brent.parquet'.
    If missing/empty, fallback to the most recent previous snapshot.
    """
    current_path = f"s3://{bucket}/{base_prefix}snapshot_date={snapshot_date}/{series_name}.parquet"

    print(f"[INFO] Attempting to load {series_name} from {current_path}")

    try:
        df = wr.s3.read_parquet(current_path)
        if not df.empty:
            return df
        print(f"[WARN] {series_name} empty, trying fallback...")
    except Exception as e:
        print(f"[WARN] Could not load {series_name} from current snapshot: {e}")

    # Fallback path discovery
    latest_prefix = find_latest_snapshot(bucket, base_prefix)
    if not latest_prefix:
        print(f"[ERROR] No fallback snapshot exists for {series_name}")
        return pd.DataFrame()

    fallback_path = f"s3://{bucket}/{latest_prefix}{series_name}.parquet"
    print(f"[INFO] Trying fallback: {fallback_path}")

    try:
        df2 = wr.s3.read_parquet(fallback_path)
        print(f"[INFO] Loaded fallback {series_name} ({len(df2)} rows)")
        return df2
    except Exception as e:
        print(f"[ERROR] Failed fallback for {series_name}: {e}")
        return pd.DataFrame()

# ===============================================================
# Helpers: FRED/EIA fetch
# ===============================================================

def fred_series(series_id: str) -> pd.DataFrame:
    """
    Stable FRED download function.
    Returns DataFrame with date, value (float), datetime-parsed.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(
        url,
        params={
            "series_id": series_id,
            "api_key": fred_key,
            "file_type": "json"
        }
    )
    r.raise_for_status()

    obs = r.json().get("observations", [])
    if not obs:
        print(f"[WARN] FRED returned no observations for {series_id}")
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(obs)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "value"])

    return df

def eia_series(series_id: str) -> pd.DataFrame:
    """
    Call EIA v2 series endpoint and return a DataFrame with columns ['date', 'value'].
    Any HTTP / parsing issues are handled gracefully by returning an empty DataFrame,
    so that S3 fallback logic can take over.
    """
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    try:
        r = requests.get(url, params={"api_key": eia_key})
    except Exception as e:
        print(f"[ERROR] EIA request failed for series {series_id}: {e}")
        # Empty DF triggers fallback_from_latest_snapshot later
        return pd.DataFrame(columns=["date", "value"])

    if r.status_code != 200:
        print(
            f"[ERROR] EIA HTTP {r.status_code} for series {series_id}; "
            f"body={r.text[:200]!r}. Returning empty DataFrame so S3 fallback can be used."
        )
        return pd.DataFrame(columns=["date", "value"])

    try:
        payload = r.json()
    except Exception as e:
        print(f"[ERROR] Unable to parse EIA JSON for series {series_id}: {e}")
        return pd.DataFrame(columns=["date", "value"])

    data = payload.get("response", {}).get("data", [])
    if not data:
        print(f"[WARN] EIA returned no data rows for series {series_id}.")
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(data)

    # Handle both shapes: ['period', 'value'] or ['date', 'value']
    if "period" in df.columns and "value" in df.columns:
        df = df[["period", "value"]].rename(columns={"period": "date"})
    elif "date" in df.columns and "value" in df.columns:
        df = df[["date", "value"]]
    else:
        print(
            f"[ERROR] EIA data for series {series_id} has unexpected columns: {df.columns.tolist()}. "
            "Returning empty DataFrame so fallback can be used."
        )
        return pd.DataFrame(columns=["date", "value"])

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date", "value"])
    return df

# ===============================================================
# 1) FETCH RAW SERIES (FRED + EIA)
# ===============================================================

print("[INFO] Fetching FX from FRED (DEXUSEU)...")
fx_raw = fred_series("DEXUSEU")          # USD per EUR

print("[INFO] Fetching Brent from EIA (PET.RBRTE.D)...")
brent_raw = eia_series("PET.RBRTE.D")

print("[INFO] Fetching Diesel from EIA (PET.EMD_EPD2D_PTE_NUS_DPG.D)...")
diesel_raw = eia_series("PET.EMD_EPD2D_PTE_NUS_DPG.D")

print("[INFO] Fetching PPI indexes from FRED...")
ppi_glass  = fred_series("WPU101706")          # glass
ppi_plast  = fred_series("WPU0721")            # plastic bottles
ppi_steel  = fred_series("WPU101702")          # steel

# Combine PPI series
ppi = (
    pd.concat(
        [
            ppi_glass.set_index("date").rename(columns={"value": "ppi_glass"}),
            ppi_plast.set_index("date").rename(columns={"value": "ppi_plastic_bottles"}),
            ppi_steel.set_index("date").rename(columns={"value": "ppi_steel"}),
        ],
        axis=1
    )
    .reset_index()
)
ppi = ppi.dropna(subset=["date"], how="any")


# ===============================================================
# 2) DAILY → WEEKLY (MONDAY)
# ===============================================================

def to_monday(d):
    """Return Monday week start."""
    d = pd.to_datetime(d, errors="coerce")
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


def weekly_mean(df: pd.DataFrame, date_col: str, value_cols: list):
    """Group to Monday week_start, taking the mean for each value column."""
    if df.empty:
        return pd.DataFrame(columns=["week_start"] + value_cols)

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["week_start"] = to_monday(out[date_col])

    return (
        out.groupby("week_start", as_index=False)[value_cols]
        .mean()
        .sort_values("week_start")
    )


# FX (usd_per_eur)
fx_w = weekly_mean(fx_raw.rename(columns={"value": "usd_per_eur"}), "date", ["usd_per_eur"])

# Brent
brent_w = weekly_mean(brent_raw.rename(columns={"value": "brent_usd_per_bbl"}), "date", ["brent_usd_per_bbl"])

# Diesel
diesel_w = weekly_mean(diesel_raw.rename(columns={"value": "diesel_usd_per_gal"}), "date", ["diesel_usd_per_gal"])

# PPI
ppi_w = weekly_mean(
    ppi.rename(columns={"date": "date"}),
    "date",
    ["ppi_glass", "ppi_plastic_bottles", "ppi_steel"]
)

print(
    "[INFO] Weekly aggregated row counts:",
    f"fx={len(fx_w)}",
    f"brent={len(brent_w)}",
    f"diesel={len(diesel_w)}",
    f"ppi={len(ppi_w)}",
)


# ===============================================================
# 3) APPLY FALLBACK (uses load_or_fallback)
# ===============================================================
base_prefix = "processed/macros/"

fx_w     = fx_w     if not fx_w.empty     else load_or_fallback("fx",     bucket, snapshot_date, base_prefix)
brent_w  = brent_w  if not brent_w.empty  else load_or_fallback("brent",  bucket, snapshot_date, base_prefix)
diesel_w = diesel_w if not diesel_w.empty else load_or_fallback("diesel", bucket, snapshot_date, base_prefix)
ppi_w    = ppi_w    if not ppi_w.empty    else load_or_fallback("ppi",    bucket, snapshot_date, base_prefix)

print("[INFO] After fallback:",
      f"fx={len(fx_w)}",
      f"brent={len(brent_w)}",
      f"diesel={len(diesel_w)}",
      f"ppi={len(ppi_w)}",
)

# ===============================================================
# 4) WRITE PER-SERIES PARQUET
# ===============================================================

dest = f"s3://{bucket}/{base_prefix}snapshot_date={snapshot_date}/"

if not fx_w.empty:
    print(f"[INFO] Writing fx.parquet ({len(fx_w)} rows)")
    wr.s3.to_parquet(fx_w, dest + "fx.parquet", index=False)

if not brent_w.empty:
    print(f"[INFO] Writing brent.parquet ({len(brent_w)} rows)")
    wr.s3.to_parquet(brent_w, dest + "brent.parquet", index=False)

if not diesel_w.empty:
    print(f"[INFO] Writing diesel.parquet ({len(diesel_w)} rows)")
    wr.s3.to_parquet(diesel_w, dest + "diesel.parquet", index=False)
else:
    print("[WARN] diesel_w empty — no diesel.parquet created.")

if not ppi_w.empty:
    print(f"[INFO] Writing ppi.parquet ({len(ppi_w)} rows)")
    wr.s3.to_parquet(ppi_w, dest + "ppi.parquet", index=False)

# ===============================================================
# Build full weekly macro grid → macros.parquet
# ===============================================================

print("[INFO] Building full weekly macro grid for macros.parquet ...")

macro_dfs = []

# Only add non-empty series; keep names stable
if not fx_w.empty:
    macro_dfs.append(fx_w[["week_start", "usd_per_eur"]])

if not brent_w.empty:
    macro_dfs.append(brent_w[["week_start", "brent_usd_per_bbl"]])

if not diesel_w.empty:
    macro_dfs.append(diesel_w[["week_start", "diesel_usd_per_gal"]])

if not ppi_w.empty:
    macro_dfs.append(
        ppi_w[["week_start", "ppi_glass", "ppi_plastic_bottles", "ppi_steel"]]
    )

if not macro_dfs:
    print("[WARN] No macro series available; skipping macros.parquet full grid.")
else:
    # Outer-merge all macro series on week_start
    macros = reduce(
        lambda left, right: pd.merge(left, right, on="week_start", how="outer"),
        macro_dfs,
    )

    # Ensure week_start is datetime for the grid construction
    macros["week_start"] = pd.to_datetime(macros["week_start"], errors="coerce")
    macros = macros.dropna(subset=["week_start"]).sort_values("week_start")

    if macros.empty:
        print("[WARN] Merged macros DataFrame is empty after cleaning; skipping macros.parquet.")
    else:
        min_w = macros["week_start"].min()
        max_w = macros["week_start"].max()
        print(f"[INFO] macros merged weekly range: {min_w} → {max_w} (rows={len(macros)})")

        # Build full Monday-week range and join
        full_weeks = pd.date_range(start=min_w, end=max_w, freq="W-MON")
        grid = pd.DataFrame({"week_start": full_weeks})

        macros = grid.merge(macros, on="week_start", how="left")

        # Forward-fill all metric columns so there are no gaps in the grid
        value_cols = [c for c in macros.columns if c != "week_start"]
        if value_cols:
            macros[value_cols] = macros[value_cols].ffill()

        # Convert week_start → epoch-ms int to stay consistent with existing parquet convention
        #    datetime64[ns] → int64 ns → ms
        if np.issubdtype(macros["week_start"].dtype, np.datetime64):
            macros["week_start"] = (macros["week_start"].view("int64") // 10**6).astype("int64")

        print(f"[INFO] Final macros grid rows: {len(macros)}")

        # Write macros.parquet alongside per-series files
        macros_dest = f"s3://{bucket}/processed/macros/snapshot_date={snapshot_date}/macros.parquet"
        print(f"[INFO] Writing macros.parquet to {macros_dest}")
        wr.s3.to_parquet(macros, macros_dest, index=False)

        print("[INFO] macros.parquet written successfully.")

print(f"[INFO] Macro ingest complete for snapshot_date={snapshot_date}")