import argparse
from datetime import date

import pandas as pd
import numpy as np
import awswrangler as wr

from olive_utils import duty_for_row  # existing helper

# ===============================================================
# Helper: epoch-ms to datetime, datetime/strings
# ===============================================================
def to_dt_ms(series: pd.Series) -> pd.Series:
    s = pd.Series(series)

    # Already datetime
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s)

    # Numeric → epoch-ms
    if np.issubdtype(s.dtype, np.number):
        return pd.to_datetime(s, unit="ms", errors="coerce")

    # String epoch-ms?
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return pd.to_datetime(s_num, unit="ms", errors="coerce")

    # Fallback: general parser
    return pd.to_datetime(s, errors="coerce")

# ===============================================================
# Arg parsing
# ===============================================================
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--BUCKET")
parser.add_argument("--SNAPSHOT_DATE")
parser.add_argument("--REGION")
parser.add_argument("--DEBUG", default="0")  # optional, non-breaking
args, _unknown = parser.parse_known_args()

bucket   = args.BUCKET
snapshot = args.SNAPSHOT_DATE or date.today().isoformat()
region   = args.REGION or "us-east-1"
DEBUG    = str(args.DEBUG).lower() in ("1", "true", "yes", "y")

if not bucket:
    raise SystemExit("Missing --BUCKET")

print(f"[INFO] build_weekly_features_v3 starting")
print(f"[INFO] bucket={bucket} snapshot_date={snapshot} region={region} DEBUG={DEBUG}")

# ===============================================================
# Input & Output paths
# ===============================================================
panel_path = f"s3://{bucket}/curated/weekly_panel/snapshot_date={snapshot}/weekly_panel.parquet"

out_prefix = f"s3://{bucket}/features/weekly_panel/snapshot_date={snapshot}/"
out_path   = out_prefix + "features.parquet"

print("[INFO] Reading weekly panel from", panel_path)
df = wr.s3.read_parquet(panel_path)

if df.empty:
    raise SystemExit("[ERROR] weekly_panel is empty; cannot build features.")

print(f"[INFO] Loaded weekly_panel rows={len(df)} cols={len(df.columns)}")

if DEBUG:
    print("[DEBUG] weekly_panel columns:")
    for c in df.columns:
        print("  -", c)

# ===============================================================
# Step 1: Timestamp normalization (epoch-ms aware)
# ===============================================================
for tcol in ["week_start", "date"]:
    if tcol in df.columns:
        df[tcol] = to_dt_ms(df[tcol])

# ===============================================================
# Step 2: Ensure snapshot_date is present & consistent
# ===============================================================
if "snapshot_date" not in df.columns:
    df["snapshot_date"] = snapshot
else:
    df["snapshot_date"] = df["snapshot_date"].fillna(snapshot).astype(str)

# ===============================================================
# Step 3: Core column sanity checks
# ===============================================================
required_cols = ["week_start", "country", "market", "grade", "price_eur_per_l"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit("[ERROR] weekly_panel missing required columns: {}".format(missing))

# grade_norm: keep existing if present, otherwise derive
if "grade_norm" not in df.columns:
    df["grade_norm"] = (
        df["grade"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

# hs_prefix: keep as-is; features job does not change tariff mapping
if "hs_prefix" not in df.columns:
    df["hs_prefix"] = np.nan

# ===============================================================
# Step 4: Coerce numeric columns (NULL-safe)
# ===============================================================
numeric_cols_panel = [
    "price_eur_per_l",
    "price_usd_per_l",
    "base_usd_per_l",
    "adval_pct",
    "specific_usd_per_kg",
    "duty_usd_per_l",
    "duty_specific_usd_per_l",
    "duty_cost",
    "usd_per_eur",
    "brent_usd_per_bbl",
    "diesel_usd_per_gal",
    "ppi_glass",
    "ppi_plastic_bottles",
    "ppi_steel",
    "ocean_proxy",
    "ocean_idx",
    "ocean_uplift",
    "diesel_uplift",
    "pack_cost",
    "deliv_hat_usd_per_l",
    "z_base",
]

for col in numeric_cols_panel:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ===============================================================
# Step 5: Usability flags – DO NOT drop rows for macro NULLs
# ===============================================================
has_core = (
    df["week_start"].notna()
    & df["country"].notna()
    & df["market"].notna()
    & df["grade_norm"].notna()
    & df["price_eur_per_l"].notna()
)

# Diagnostics (not used to filter)
has_fx        = df["usd_per_eur"].notna() if "usd_per_eur" in df.columns else False
has_price_usd = df["price_usd_per_l"].notna() if "price_usd_per_l" in df.columns else False
has_tariff    = (
    (df["adval_pct"].notna() | df["specific_usd_per_kg"].notna())
    if "adval_pct" in df.columns and "specific_usd_per_kg" in df.columns
    else False
)
has_duty_usd  = df["duty_usd_per_l"].notna() if "duty_usd_per_l" in df.columns else False
has_ocean     = df["ocean_proxy"].notna() if "ocean_proxy" in df.columns else False
has_diesel    = df["diesel_usd_per_gal"].notna() if "diesel_usd_per_gal" in df.columns else False
has_ppi       = (
    (
        df.get("ppi_glass", pd.Series(index=df.index)).notna()
        | df.get("ppi_plastic_bottles", pd.Series(index=df.index)).notna()
        | df.get("ppi_steel", pd.Series(index=df.index)).notna()
    )
    if any(c in df.columns for c in ["ppi_glass", "ppi_plastic_bottles", "ppi_steel"])
    else False
)

if DEBUG:
    print("[DEBUG] has_core rows:", has_core.sum(), " / ", len(df))
    print("[DEBUG] has_fx rows:", has_fx.sum() if hasattr(has_fx, "sum") else has_fx)
    print("[DEBUG] has_tariff rows:", has_tariff.sum() if hasattr(has_tariff, "sum") else has_tariff)
    print("[DEBUG] has_ocean rows:", has_ocean.sum() if hasattr(has_ocean, "sum") else has_ocean)
    print("[DEBUG] has_ppi rows:", has_ppi.sum() if hasattr(has_ppi, "sum") else has_ppi)

# CRITICAL: only core identifiers & price considered; macros may be NULL
mask_usable = has_core

df_usable = df[mask_usable].copy()

print(f"[INFO] Usable rows for features: {len(df_usable)} / {len(df)}")

if df_usable.empty:
    raise SystemExit("[ERROR] No usable rows for features after core filtering.")

# ===============================================================
# Step 6: Impute price_usd_per_l if FX is present
# ===============================================================
if "price_usd_per_l" not in df_usable.columns:
    df_usable["price_usd_per_l"] = np.nan

if "usd_per_eur" in df_usable.columns:
    mask_price_impute = (
        df_usable["price_usd_per_l"].isna()
        & df_usable["price_eur_per_l"].notna()
        & df_usable["usd_per_eur"].notna()
    )
    if mask_price_impute.any():
        print(f"[INFO] Imputing price_usd_per_l for {mask_price_impute.sum()} rows...")
        df_usable.loc[mask_price_impute, "price_usd_per_l"] = (
            df_usable.loc[mask_price_impute, "price_eur_per_l"]
            * df_usable.loc[mask_price_impute, "usd_per_eur"]
        )

# ===============================================================
# Step 7: Ensure base_usd_per_l exists (mirror panel semantics)
# ===============================================================
if "base_usd_per_l" not in df_usable.columns:
    df_usable["base_usd_per_l"] = df_usable["price_usd_per_l"]
else:
    df_usable["base_usd_per_l"] = df_usable["base_usd_per_l"].fillna(
        df_usable["price_usd_per_l"]
    )

# ===============================================================
# Step 8: Impute duty_usd_per_l if possible
# ===============================================================
if "duty_usd_per_l" not in df_usable.columns:
    df_usable["duty_usd_per_l"] = np.nan

if "adval_pct" in df_usable.columns and "specific_usd_per_kg" in df_usable.columns:
    mask_duty_impute = (
        df_usable["duty_usd_per_l"].isna()
        & df_usable["base_usd_per_l"].notna()
        & (df_usable["adval_pct"].notna() | df_usable["specific_usd_per_kg"].notna())
    )

    if mask_duty_impute.any():
        print(f"[INFO] Imputing duty_usd_per_l for {mask_duty_impute.sum()} rows via duty_for_row")
        df_usable.loc[mask_duty_impute, "duty_usd_per_l"] = df_usable.loc[
            mask_duty_impute
        ].apply(
            lambda r: duty_for_row(
                r["base_usd_per_l"],
                r["grade_norm"],
                r["specific_usd_per_kg"] if pd.notna(r["specific_usd_per_kg"]) else 0.0,
                r["adval_pct"] if pd.notna(r["adval_pct"]) else 0.0,
            ),
            axis=1,
        )

# ===============================================================
# Step 9: Pack defaults (if any rows are missing)
# ===============================================================
if "pack" not in df_usable.columns:
    df_usable["pack"] = "glass"

PACK_COST_TABLE = {
    "glass": 0.22,
    "plastic": 0.12,
    "steel": 0.30,
}

if "pack_cost" not in df_usable.columns:
    df_usable["pack_cost"] = df_usable["pack"].map(PACK_COST_TABLE).fillna(0.22)
else:
    # fill missing using default mapping
    missing_pack_cost = df_usable["pack_cost"].isna()
    if missing_pack_cost.any():
        df_usable.loc[missing_pack_cost, "pack_cost"] = (
            df_usable.loc[missing_pack_cost, "pack"]
            .map(PACK_COST_TABLE)
            .fillna(0.22)
        )

# ===============================================================
# Step 10: Delivered price & z_base safety
# ===============================================================
if "deliv_hat_usd_per_l" not in df_usable.columns:
    # recompute if panel didn't
    print("[WARN] deliv_hat_usd_per_l missing – recomputing from components.")
    df_usable["deliv_hat_usd_per_l"] = (
        df_usable["base_usd_per_l"].fillna(0.0)
        + df_usable.get("pack_cost", 0.0).fillna(0.0)
        + df_usable.get("ocean_uplift", 0.0).fillna(0.0)
        + df_usable.get("diesel_uplift", 0.0).fillna(0.0)
        + df_usable.get("duty_cost", 0.0).fillna(0.0)
    )

if "z_base" not in df_usable.columns:
    base_mean = df_usable["base_usd_per_l"].astype(float).mean()
    base_std  = df_usable["base_usd_per_l"].astype(float).std()
    if not base_std or np.isnan(base_std):
        df_usable["z_base"] = 0.0
    else:
        df_usable["z_base"] = (
            df_usable["base_usd_per_l"].astype(float) - base_mean
        ) / base_std

# ===============================================================
# Step 11: Define feature column sets
# ===============================================================
base_feature_cols = [
    # Identifiers
    "week_start",
    "country",
    "market",
    "grade",
    "grade_norm",

    # Prices & FX
    "price_eur_per_l",
    "price_usd_per_l",

    # Tariffs
    "hs_prefix",
    "adval_pct",
    "specific_usd_per_kg",
    "duty_usd_per_l",

    # Macros for model
    "usd_per_eur",
    "brent_usd_per_bbl",
    "diesel_usd_per_gal",
    "ppi_glass",
    "ppi_plastic_bottles",
    "ppi_steel",
    "ocean_proxy",

    # Snapshot
    "snapshot_date",
]

# Enrichments (only kept if present)
extra_notebook_cols = [
    "iso2",
    "pack",
    "pack_cost",
    "ocean_idx",
    "ocean_uplift",
    "diesel_uplift",
    "duty_specific_usd_per_l",
    "duty_cost",
    "duty_rate",                 # same as adval_pct, but included
    "base_usd_per_l",
    "deliv_hat_usd_per_l",
    "z_base",
]

extra_notebook_cols = [c for c in extra_notebook_cols if c in df_usable.columns]

feature_cols = base_feature_cols + extra_notebook_cols
feature_cols = [c for c in feature_cols if c in df_usable.columns]

# Also include any remaining columns from the weekly panel so we never
# silently drop new fields (non-breaking mirror of weekly_panel schema).
residual_cols = [
    c for c in df_usable.columns
    if c not in feature_cols and not c.startswith("_")
]
final_cols = feature_cols + residual_cols

print("[INFO] Final feature column set (ordered):")
for c in final_cols:
    print("  -", c)

# ===============================================================
# Step 12: Build features DF, sort, enforce numeric dtypes
# ===============================================================
features = df_usable[final_cols].sort_values(
    ["week_start", "country", "market", "grade"]
)

# Enforce float64 on key numeric fields (backward compatibility)
numeric_cols_final = [
    "price_eur_per_l",
    "price_usd_per_l",
    "adval_pct",
    "specific_usd_per_kg",
    "duty_usd_per_l",
    "usd_per_eur",
    "brent_usd_per_bbl",
    "diesel_usd_per_gal",
    "ppi_glass",
    "ppi_plastic_bottles",
    "ppi_steel",
    "ocean_proxy",
    "pack_cost",
    "duty_specific_usd_per_l",
    "duty_cost",
    "base_usd_per_l",
    "deliv_hat_usd_per_l",
    "ocean_uplift",
    "diesel_uplift",
    "z_base",
]

for col in numeric_cols_final:
    if col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce").astype("float64")

print(f"[INFO] Final features rows={len(features)} cols={len(features.columns)}")

for col in ["usd_per_eur", "brent_usd_per_bbl", "diesel_usd_per_gal",
            "duty_usd_per_l", "deliv_hat_usd_per_l", "z_base"]:
    if col in features.columns:
        non_null = features[col].notna().sum()
        print(f"[QA] {col}: non-null {non_null} / {len(features)}")
        
# ===============================================================
# Step 13: Write features to S3 (same path/filename as v2)
# ===============================================================
print("[INFO] Writing features to", out_path)
wr.s3.to_parquet(features, out_path, dataset=False, index=False)

print("[INFO] Features written OK.")

# ===============================================================
# Model-Ready Minimal Feature Set
# ===============================================================

model_cols = [
    "week_start",
    "country",
    "market",
    "grade",
    "grade_norm",
    "price_eur_per_l",
    "price_usd_per_l",
    "hs_prefix",
    "adval_pct",
    "specific_usd_per_kg",
    "duty_usd_per_l",
    "usd_per_eur",
    "brent_usd_per_bbl",
    "diesel_usd_per_gal",
    "ppi_glass",
    "ppi_plastic_bottles",
    "ppi_steel",
    "ocean_proxy",
    "snapshot_date",
]

# Only keep columns that actually exist (safe forward/backward compatibility)
model_cols = [c for c in model_cols if c in df_usable.columns]

model_features = df_usable[model_cols].copy()

# Sort the model output for consistency
model_features = model_features.sort_values(
    ["week_start", "country", "market", "grade"]
)

# Output path for model-ready features
model_out_path = (
    f"s3://{bucket}/features/weekly_panel/snapshot_date={snapshot}/"
    "model_features.parquet"
)

print("[INFO] Writing model-ready features to", model_out_path)
wr.s3.to_parquet(
    df=model_features,
    path=model_out_path,
    dataset=False,
    index=False,
)
print("[INFO] Model-ready features written OK.")

# ===============================================================
# DATA QUALITY SUMMARY (printed to CloudWatch)
# ===============================================================

print("\n[SUMMARY] --- DATASET OVERVIEW ---")
print(f"Total rows     : {len(df)}")
print(f"Usable rows    : {len(df_usable)}")
print(f"Feature rows   : {len(features)}")

# Unique entity counts
for col in ["country", "market", "grade_norm"]:
    if col in df_usable.columns:
        print(f"Unique {col:<12}: {df_usable[col].nunique()}")

# Non-null coverage for key model fields
key_fields = [
    "usd_per_eur", "brent_usd_per_bbl", "diesel_usd_per_gal",
    "ocean_proxy", "ppi_glass", "ppi_plastic_bottles", "ppi_steel",
    "duty_usd_per_l", "deliv_hat_usd_per_l"
]

print("\n[SUMMARY] --- NON-NULL COVERAGE ---")
for col in key_fields:
    if col in df_usable.columns:
        pct = df_usable[col].notna().mean() * 100
        print(f"{col:<25}: {pct:5.1f}% non-null")

# Range summaries for major numeric features
numeric_checks = [
    "price_eur_per_l", "price_usd_per_l",
    "brent_usd_per_bbl", "diesel_usd_per_gal",
    "usd_per_eur", "duty_usd_per_l",
    "ocean_proxy", "deliv_hat_usd_per_l"
]

print("\n[SUMMARY] --- NUMERIC RANGES ---")
for col in numeric_checks:
    if col in df_usable.columns:
        print(f"{col:<25}: min={df_usable[col].min():.3f}, max={df_usable[col].max():.3f}")

if "week_start" in df_usable.columns:
    print("\n[SUMMARY] --- TIME COVERAGE ---")
    print("Earliest week:", df_usable["week_start"].min())
    print("Latest week  :", df_usable["week_start"].max())
    print("Total weeks  :", df_usable["week_start"].nunique())

print("\n[SUMMARY] --- DISTRIBUTION SNAPSHOTS ---")
for col in ["price_usd_per_l", "deliv_hat_usd_per_l", "z_base"]:
    if col in df_usable.columns:
        desc = df_usable[col].describe()
        print(f"\n{col} distribution:\n{desc}")

print("[INFO] weekly_features job completed successfully.")