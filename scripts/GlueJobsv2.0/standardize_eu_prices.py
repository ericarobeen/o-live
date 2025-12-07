import os, argparse, re
from datetime import date
import pandas as pd
import awswrangler as wr

# from your shared utils (Python 3.7-safe version)
from olive_utils import canon_grade, eur100kg_to_eur_per_l, s3_latest_key

# ============
# Arg parsing (robust: ignores unknown Glue flags)
# ============
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--BUCKET")
parser.add_argument("--RAW_PREFIX")
parser.add_argument("--SNAPSHOT_DATE")
parser.add_argument("--REGION")
# Optional override: how to interpret the 'price' column if units are ambiguous
# Allowed: per_l, per_kg, per_100kg (default)
parser.add_argument("--PRICE_UNIT")
args, _unknown = parser.parse_known_args()

bucket     = args.BUCKET
raw_prefix = (args.RAW_PREFIX or "raw/eu_prices/").strip("/")
snapshot   = args.SNAPSHOT_DATE or date.today().isoformat()
region     = args.REGION or "us-east-1"
price_unit = (args.PRICE_UNIT or "per_100kg").lower()

print(f"[INFO] Using snapshot_date={snapshot}")

if not bucket:
    raise SystemExit("Missing --BUCKET")

print(f"[INFO] bucket={bucket} raw_prefix={raw_prefix} snapshot={snapshot} region={region} price_unit={price_unit}")

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
# Locate and read latest file
# ============
key = s3_latest_key(bucket, raw_prefix + "/", suffixes=(".xlsx", ".xls", ".csv"))
path = f"s3://{bucket}/{key}"
print("[INFO] Reading", path)

if key.lower().endswith((".xlsx", ".xls")):
    # Try a sheet named "Data", else first sheet
    try:
        df_raw = wr.s3.read_excel(path, sheet_name="Data")
    except Exception:
        df_raw = wr.s3.read_excel(path, sheet_name=0)
else:
    df_raw = wr.s3.read_csv(path)

if df_raw.empty:
    raise SystemExit("[ERROR] EU prices file is empty.")

# Normalize headers
orig_cols = list(df_raw.columns)
df_raw.columns = _norm_cols(df_raw.columns)
cols = list(df_raw.columns)
print("[INFO] Columns (normalized):", cols)

# ============
# Column selection
# ============
# Date candidates (your file shows 'referencefrom' and also 'year' + 'week')
date_from_col = _pick_col(["referencefrom", "reference_from", "ref_from", "date"], cols)
year_col      = _pick_col(["year", "yr"], cols)
week_col      = _pick_col(["week", "wk"], cols)

# Identity columns
country_col = _pick_col(["member_state", "memberstate", "country"], cols)
market_col  = _pick_col(["market", "city", "location"], cols)
grade_col   = _pick_col(["category", "product", "prod", "grade", "variety"], cols)

# Price column: your file has 'price'
price_col = _pick_col(["price", "price_eur_per_l", "price_eur_per_kg", "eur_per_l", "eur_per_kg", "eur_100kg", "price_eur_per_100kg"], cols)
if not price_col:
    raise ValueError(f"Missing a recognizable price column. Columns: {cols}")

# ============
# Build working frame
# ============
d = df_raw.copy()

# Date derivation
if date_from_col:
    dt = pd.to_datetime(d[date_from_col], errors="coerce")
elif year_col and week_col:
    dt = _iso_week_start(d[year_col], d[week_col])
else:
    raise ValueError("Could not determine a date column (looked for referencefrom or year+week).")

d["date"] = dt.dt.tz_localize(None)
print("[INFO] Sample dates:", d["date"].dropna().head(3).tolist())

# Price to €/L
# Priority:
#   1) If the column is clearly a €/L or €/kg column by name, use that mapping.
#   2) Else use explicit --PRICE_UNIT if provided.
#   3) Else default to €/100kg → €/L (common for EU price sheets).
name_l = price_col.lower()
if name_l in ("price_eur_per_l", "eur_per_l"):
    d["price_eur_per_l"] = pd.to_numeric(d[price_col], errors="coerce")
elif name_l in ("price_eur_per_kg", "eur_per_kg"):
    d["price_eur_per_l"] = pd.to_numeric(d[price_col], errors="coerce") / 0.916
elif name_l in ("eur_100kg", "price_eur_per_100kg"):
    d["price_eur_per_l"] = pd.to_numeric(d[price_col], errors="coerce").apply(eur100kg_to_eur_per_l)
else:
    # ambiguous 'price' header: follow --PRICE_UNIT (default per_100kg)
    pv = pd.to_numeric(d[price_col], errors="coerce")
    if price_unit == "per_l":
        d["price_eur_per_l"] = pv
    elif price_unit == "per_kg":
        d["price_eur_per_l"] = pv / 0.916
    else:
        d["price_eur_per_l"] = pv.apply(eur100kg_to_eur_per_l)

# Grade / Country / Market
if grade_col and grade_col in d.columns:
    d["grade"] = d[grade_col].apply(canon_grade)
else:
    d["grade"] = None

d["country"] = d[country_col] if country_col else None
d["market"]  = d[market_col]  if market_col  else None

# ============
# Final frame & write
# ============
out = d[["date", "country", "market", "grade", "price_eur_per_l"]].dropna(subset=["date", "price_eur_per_l"])
out = out.sort_values("date")

if out.empty:
    raise SystemExit("[ERROR] No valid rows after parsing date and price. Check input file/units.")

dest = f"s3://{bucket}/processed/eu_prices/snapshot_date={snapshot}/eu_prices.parquet"
print("[INFO] Writing", dest)
wr.s3.to_parquet(out, dest, dataset=False, index=False)
print("[INFO] EU prices written OK.")