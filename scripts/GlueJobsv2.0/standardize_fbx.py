import os, argparse, re
from datetime import date

import pandas as pd
import awswrangler as wr

from olive_utils import s3_latest_key, to_monday_week

# ============
# Arg parsing (robust: ignores unknown Glue flags)
# ============
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--BUCKET")
parser.add_argument("--RAW_PREFIX")
parser.add_argument("--SNAPSHOT_DATE")
parser.add_argument("--REGION")
args, _unknown = parser.parse_known_args()

bucket     = args.BUCKET
# default now points to raw/fbx_index/
raw_prefix = (args.RAW_PREFIX or "raw/fbx_index/").strip("/")
snapshot   = args.SNAPSHOT_DATE or date.today().isoformat()
region     = args.REGION or "us-east-1"

print(f"[INFO] Using snapshot_date={snapshot}")

if not bucket:
    raise SystemExit("Missing --BUCKET")

print(f"[INFO] bucket={bucket} raw_prefix={raw_prefix} snapshot={snapshot} region={region}")

# ============
# Helpers
# ============
def _norm_cols(cols):
    """snake_case + lowercase; keep letters/digits/_"""
    return [re.sub(r"\W+", "_", str(c).strip().lower()) for c in cols]

def _pick_col(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

def _iso_week_start(year_series, week_series):
    """Return Monday dates from ISO year/week."""
    y = pd.to_numeric(year_series, errors="coerce").astype("Int64")
    w = pd.to_numeric(week_series, errors="coerce").astype("Int64")
    s = y.astype(str).str.zfill(4) + "-" + w.astype(str).str.zfill(2) + "-1"
    # %G = ISO year, %V = ISO week number, %u = ISO weekday (1=Mon)
    return pd.to_datetime(s, format="%G-%V-%u", errors="coerce")

# ============
# Locate and read latest FBX file
# ============
key = s3_latest_key(bucket, raw_prefix + "/", suffixes=(".csv", ".xlsx", ".xls"))
path = f"s3://{bucket}/{key}"
print("[INFO] Reading", path)

if key.lower().endswith((".xlsx", ".xls")):
    # FBX is usually CSV, but support Excel just in case
    try:
        df_raw = wr.s3.read_excel(path, sheet_name=0)
    except Exception as e:
        raise SystemExit(f"Failed to read FBX Excel file: {e}")
else:
    df_raw = wr.s3.read_csv(path)

if df_raw.empty:
    raise SystemExit("[ERROR] FBX file is empty.")

# Normalize headers
orig_cols = list(df_raw.columns)
df_raw.columns = _norm_cols(df_raw.columns)
cols = list(df_raw.columns)
print("[INFO] Columns (normalized):", cols)

# ============
# Column selection
# ============
# Date-like columns
date_col      = _pick_col(["date", "observation_date", "day", "dt"], cols)
year_col      = _pick_col(["year", "yr"], cols)
week_col      = _pick_col(["week", "wk"], cols)

# FBX index column (varies by provider)
# >>> This is the important line: include 'fbx_global'
fbx_col = _pick_col(
    ["fbx_global", "fbx", "index", "fbx_index", "price", "value"],
    cols
)
if not fbx_col:
    raise ValueError(f"Missing a recognizable FBX column. Columns: {cols}")

# ============
# Build working frame
# ============
d = df_raw.copy()

# Date derivation
if date_col:
    dt = pd.to_datetime(d[date_col], errors="coerce")
elif year_col and week_col:
    dt = _iso_week_start(d[year_col], d[week_col])
else:
    raise ValueError("Could not determine a date column (looked for date or year+week).")

d["date"] = dt.dt.tz_localize(None)
print("[INFO] Sample dates:", d["date"].dropna().head(3).tolist())

# FBX index as float
d["fbx"] = pd.to_numeric(d[fbx_col], errors="coerce")

# Drop rows with no date or no index
d = d.dropna(subset=["date", "fbx"])
if d.empty:
    raise SystemExit("[ERROR] No valid rows after parsing date and fbx index.")

# Align to Monday weeks
d["week_start"] = to_monday_week(d["date"])
wk = (
    d.groupby("week_start", as_index=False)["fbx"]
     .mean()
     .sort_values("week_start")
)

if wk.empty:
    raise SystemExit("[ERROR] No weekly FBX data after aggregation.")

# ============
# Write to processed layer
# ============
dest_prefix = f"s3://{bucket}/processed/fbx/snapshot_date={snapshot}/"
dest_path   = dest_prefix + "fbx.parquet"
print("[INFO] Writing", dest_path)
wr.s3.to_parquet(wk, dest_path, dataset=False, index=False)
print("[INFO] FBX written OK.")