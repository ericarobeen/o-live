import argparse, re
from datetime import date

import pandas as pd
import awswrangler as wr

from olive_utils import s3_latest_key

# ============
# Arg parsing (robust: ignores unknown Glue flags)
# ============
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--BUCKET")
parser.add_argument("--RAW_PREFIX")
parser.add_argument("--SNAPSHOT_DATE")   # not strictly needed, but kept for consistency
parser.add_argument("--REGION")
args, _unknown = parser.parse_known_args()

bucket     = args.BUCKET
raw_prefix = (args.RAW_PREFIX or "raw/tariffs/reference/").strip("/")
snapshot   = args.SNAPSHOT_DATE or date.today().isoformat()
region     = args.REGION or "us-east-1"

print(f"[INFO] Using snapshot_date={snapshot}")

if not bucket:
    raise SystemExit("Missing --BUCKET")

print("[INFO] bucket={}, raw_prefix={}, snapshot={}, region={}".format(
    bucket, raw_prefix, snapshot, region
))

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

# ============
# Locate and read latest tariffs file
# ============
key = s3_latest_key(bucket, raw_prefix + "/", suffixes=(".xlsx", ".xls", ".csv"))
path = "s3://{}/{}".format(bucket, key)
print("[INFO] Reading", path)

if key.lower().endswith((".xlsx", ".xls")):
    df_raw = wr.s3.read_excel(path, sheet_name=0)
else:
    df_raw = wr.s3.read_csv(path)

if df_raw.empty:
    raise SystemExit("[ERROR] Tariffs file is empty.")

orig_cols = list(df_raw.columns)
df_raw.columns = _norm_cols(df_raw.columns)
cols = list(df_raw.columns)
print("[INFO] Columns (normalized):", cols)

# ============
# Column mapping tailored to your file
# ============
# HS code: your file has 'hts8' as the main code column
hs_col = _pick_col(["hs_prefix", "hs4", "hs_code", "hs", "hts8"], cols)
if not hs_col:
    raise ValueError("Could not find hs_prefix-like column in {}".format(cols))

# MFN ad valorem % and specific rate
# Your file exposes them as mfn_ad_val_rate and mfn_specific_rate.
adval_col = _pick_col(
    ["mfn_ad_val_rate", "mfn_ave", "mfn_ad_val", "adval_pct", "ad_valorem_pct"],
    cols,
)
spec_col = _pick_col(
    ["mfn_specific_rate", "mfn_specific", "specific_usd_per_kg", "specific_rate"],
    cols,
)

# We don't have a direct 'grade' column in this sheet; that's fine.
grade_col = _pick_col(["grade", "product_grade", "category", "quality"], cols)

# ============
# Build standardized frame
# ============
d = df_raw.copy()

# Grade: optional; mostly None here
if grade_col and grade_col in d.columns:
    d["grade"] = d[grade_col].astype(str).str.strip().str.upper()
else:
    d["grade"] = None

# hs_prefix: derive HS-4 from your 'hts8' or other HS column
hs_raw = d[hs_col].astype(str).str.strip()
# keep only digits, then take first 4 characters as HS-4
hs_clean = hs_raw.str.replace(r"\D", "", regex=True)
d["hs_prefix"] = hs_clean.str.slice(0, 4)

# Ad valorem %: default 0 if missing
if adval_col and adval_col in d.columns:
    d["adval_pct"] = pd.to_numeric(d[adval_col], errors="coerce")
else:
    d["adval_pct"] = 0.0

# Specific $/kg: default 0 if missing
if spec_col and spec_col in d.columns:
    d["specific_usd_per_kg"] = pd.to_numeric(d[spec_col], errors="coerce")
else:
    d["specific_usd_per_kg"] = 0.0

out = d[["grade", "hs_prefix", "adval_pct", "specific_usd_per_kg"]].dropna(subset=["hs_prefix"])
out = out.drop_duplicates(["grade", "hs_prefix"], keep="last")

if out.empty:
    raise SystemExit("[ERROR] No tariff rows with hs_prefix after cleaning.")

dest = "s3://{}/processed/tariffs/latest/tariffs.parquet".format(bucket)
print("[INFO] Writing", dest)
wr.s3.to_parquet(out, dest, dataset=False, index=False)
print("[INFO] Tariffs written OK.")