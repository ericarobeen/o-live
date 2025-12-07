import argparse
from datetime import date

import pandas as pd
import numpy as np
import awswrangler as wr

from olive_utils import (
    to_monday_week,
    ocean_proxy,
    duty_for_row,
    GRADE_TO_HS,
)

# ===================================================================
# SAFE DATETIME CONVERSION - Epoch-ms
# ===================================================================
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

# ===================================================================
# ARGUMENT PARSING (IGNORES UNKNOWN GLUE FLAGS)
# ===================================================================
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--BUCKET")
parser.add_argument("--SNAPSHOT_DATE")
parser.add_argument("--REGION")
args, _unknown = parser.parse_known_args()

bucket   = args.BUCKET
snapshot = args.SNAPSHOT_DATE or date.today().isoformat()
region   = args.REGION or "us-east-1"

if not bucket:
    raise SystemExit("Missing --BUCKET")

print(f"[INFO] Using snapshot_date={snapshot}")
print(f"[INFO] bucket={bucket} snapshot={snapshot} region={region}")


# ===================================================================
# PATHS
# ===================================================================
macros_prefix = f"s3://{bucket}/processed/macros/snapshot_date={snapshot}/"
eu_path       = f"s3://{bucket}/processed/eu_prices/snapshot_date={snapshot}/eu_prices.parquet"
fbx_path      = f"s3://{bucket}/processed/fbx/snapshot_date={snapshot}/fbx.parquet"
tariffs_path  = f"s3://{bucket}/processed/tariffs/latest/tariffs.parquet"

fx_path     = macros_prefix + "fx.parquet"
brent_path  = macros_prefix + "brent.parquet"
diesel_path = macros_prefix + "diesel.parquet"
ppi_path    = macros_prefix + "ppi.parquet"


# ===================================================================
# LOAD INDIVIDUAL MACRO FILES
# ===================================================================
fx     = wr.s3.read_parquet(fx_path)
brent  = wr.s3.read_parquet(brent_path)
diesel = wr.s3.read_parquet(diesel_path)
ppi    = wr.s3.read_parquet(ppi_path)

print("[INFO] Loaded macro components:",
      "fx:", len(fx),
      "brent:", len(brent),
      "diesel:", len(diesel),
      "ppi:", len(ppi))


# ===================================================================
# NORMALIZE week_start FOR EACH MACRO COMPONENT
# ===================================================================
for df, name in [(fx,"fx"), (brent,"brent"), (diesel,"diesel"), (ppi,"ppi")]:
    if "week_start" not in df.columns:
        raise SystemExit(f"[ERROR] {name}.parquet missing week_start")
    df["week_start"] = to_dt_ms(df["week_start"])
    df["week_start"] = to_monday_week(df["week_start"])
    df.sort_values("week_start", inplace=True)

# ===================================================================
# FX COLUMN NORMALIZATION BEFORE MERGE
# ===================================================================
# fx may contain: usd_per_eur, value, eurusd, eur_usd, usdeur
if "usd_per_eur" not in fx.columns:
    if "value" in fx.columns:
        fx.rename(columns={"value": "usd_per_eur"}, inplace=True)
    elif "eurusd" in fx.columns:
        fx.rename(columns={"eurusd": "usd_per_eur"}, inplace=True)
    elif "eur_usd" in fx.columns:
        fx.rename(columns={"eur_usd": "usd_per_eur"}, inplace=True)
    elif "usdeur" in fx.columns:
        fx["usd_per_eur"] = 1 / fx["usdeur"]
    else:
        print("[WARN] FX missing known columns — creating placeholder usd_per_eur.")
        fx["usd_per_eur"] = np.nan

# ensure FX keeps only the needed fields
fx = fx[["week_start", "usd_per_eur"]]

# ===================================================================
# RENAME OTHER MACRO SERIES TO WELL-DEFINED NAMES
# ===================================================================
if "value" in brent.columns:
    brent.rename(columns={"value": "brent_usd_per_bbl"}, inplace=True)
if "value" in diesel.columns:
    diesel.rename(columns={"value": "diesel_usd_per_gal"}, inplace=True)

# PPI may be in melted or wide format — ensure safe compatibility
ppi.rename(columns={c: c for c in ppi.columns}, inplace=True)
for col in ["ppi_glass", "ppi_plastic_bottles", "ppi_steel"]:
    if col not in ppi.columns:
        ppi[col] = np.nan
ppi = ppi[["week_start", "ppi_glass", "ppi_plastic_bottles", "ppi_steel"]]


# ===================================================================
# MERGE INTO macros
# ===================================================================
macros = (
    fx
    .merge(brent[["week_start", "brent_usd_per_bbl"]], on="week_start", how="outer")
    .merge(diesel[["week_start", "diesel_usd_per_gal"]], on="week_start", how="outer")
    .merge(ppi, on="week_start", how="outer")
)

macros = macros.sort_values("week_start").drop_duplicates("week_start", keep="last")

print("[INFO] Final merged macros rows:", len(macros),
      "week range:", macros["week_start"].min(), "→", macros["week_start"].max())

# ===================================================================
# LOAD OTHER INPUTS
# ===================================================================
print("[INFO] Reading EU prices:", eu_path)
eu = wr.s3.read_parquet(eu_path)

print("[INFO] Reading FBX:", fbx_path)
fbx = wr.s3.read_parquet(fbx_path)

print("[INFO] Reading tariffs:", tariffs_path)
tariffs = wr.s3.read_parquet(tariffs_path)

# ===================================================================
# NORMALIZE EU + FBX week_start
# ===================================================================
eu["date"] = to_dt_ms(eu["date"])
eu["week_start"] = to_monday_week(eu["date"])
eu = eu.sort_values("week_start")

fbx["week_start"] = to_dt_ms(fbx["week_start"])
fbx["week_start"] = to_monday_week(fbx["week_start"])
fbx = fbx.sort_values("week_start")

# ===================================================================
# BUILD PANEL
# ===================================================================
panel = eu.copy()
del eu

# ensure essentials
panel["market"] = panel.get("market", None)
panel["iso2"] = panel["country"]

# ===================================================================
# PACK COST
# ===================================================================
PACK_COST_TABLE = {"glass":0.22,"plastic":0.12,"steel":0.30}
panel["pack"] = panel.get("pack", "glass")
panel["pack_cost"] = panel["pack"].map(PACK_COST_TABLE).fillna(0.22)

# ===================================================================
# FX MERGE
# ===================================================================
panel = panel.merge(macros[["week_start","usd_per_eur"]], on="week_start", how="left")
panel["price_usd_per_l"] = panel["price_eur_per_l"] * panel["usd_per_eur"]

# ===================================================================
# MERGE REMAINING MACROS
# ===================================================================
macro_cols = [
    "week_start",
    "brent_usd_per_bbl",
    "diesel_usd_per_gal",
    "ppi_glass",
    "ppi_plastic_bottles",
    "ppi_steel",
]
panel = panel.merge(macros[macro_cols], on="week_start", how="left")

# ===================================================================
# OCEAN PROXY
# ===================================================================
if "brent_usd_per_bbl" in macros.columns and "fbx" in fbx.columns:
    temp = macros[["week_start","brent_usd_per_bbl"]].merge(
        fbx[["week_start","fbx"]],
        on="week_start",
        how="outer"
    )
    temp["ocean_proxy"] = ocean_proxy(temp["fbx"], temp["brent_usd_per_bbl"])
    panel = panel.merge(temp[["week_start","ocean_proxy"]], on="week_start", how="left")
else:
    panel["ocean_proxy"] = np.nan

panel["ocean_idx"] = panel["ocean_proxy"]

# ===================================================================
# UPLIFTS
# ===================================================================
panel["ocean_uplift"] = 0.003 * panel["ocean_proxy"].astype(float)

diesel_mean = panel["diesel_usd_per_gal"].astype(float).mean()
panel["diesel_uplift"] = 0.15 * (panel["diesel_usd_per_gal"] - diesel_mean)

# ===================================================================
# TARIFFS
# ===================================================================
tariffs["hs_prefix"] = tariffs["hs_prefix"].astype(str).str.strip()
tariffs["adval_pct"] = pd.to_numeric(tariffs["adval_pct"], errors="coerce").fillna(0)
tariffs["specific_usd_per_kg"] = pd.to_numeric(tariffs["specific_usd_per_kg"], errors="coerce").fillna(0)

panel["grade_norm"] = panel["grade"].astype(str).str.upper().str.strip()
panel["hs_prefix"] = panel["grade_norm"].map(GRADE_TO_HS)

panel = panel.merge(
    tariffs[["hs_prefix","adval_pct","specific_usd_per_kg"]],
    on="hs_prefix",
    how="left"
)

panel["adval_pct"].fillna(0, inplace=True)
panel["specific_usd_per_kg"].fillna(0, inplace=True)
panel["duty_rate"] = panel["adval_pct"]

# ===================================================================
# DUTY COSTS
# ===================================================================
DENSITY = 0.916
panel["duty_specific_usd_per_l"] = (panel["specific_usd_per_kg"] * DENSITY).fillna(0)

panel["duty_cost"] = (
    (panel["adval_pct"]/100) * panel["price_usd_per_l"].astype(float)
    + panel["duty_specific_usd_per_l"]
)

panel["duty_usd_per_l"] = panel.apply(
    lambda r: duty_for_row(
        r["price_usd_per_l"],
        r["grade_norm"],
        r["specific_usd_per_kg"],
        r["adval_pct"],
    ) if pd.notna(r["price_usd_per_l"]) else None,
    axis=1
)

# ===================================================================
# DELIVERED HAT
# ===================================================================
panel["base_usd_per_l"] = panel["price_usd_per_l"]
panel["deliv_hat_usd_per_l"] = (
      panel["base_usd_per_l"].fillna(0)
    + panel["pack_cost"].fillna(0)
    + panel["ocean_uplift"].fillna(0)
    + panel["diesel_uplift"].fillna(0)
    + panel["duty_cost"].fillna(0)
)

# ===================================================================
# Z-SCORE
# ===================================================================
base_mean = panel["base_usd_per_l"].astype(float).mean()
base_std  = panel["base_usd_per_l"].astype(float).std()
panel["z_base"] = 0 if base_std == 0 else (panel["base_usd_per_l"] - base_mean) / base_std

# ===================================================================
# FINAL METADATA + SORT
# ===================================================================
panel["snapshot_date"] = snapshot

cols_order = [
    "week_start","snapshot_date","country","iso2","market","grade",
    "grade_norm","hs_prefix","pack",
    "price_eur_per_l","price_usd_per_l","base_usd_per_l","usd_per_eur",
    "adval_pct","duty_rate","specific_usd_per_kg","duty_specific_usd_per_l",
    "duty_cost","duty_usd_per_l",
    "brent_usd_per_bbl","ocean_proxy","ocean_idx","ocean_uplift",
    "diesel_usd_per_gal","diesel_uplift","pack_cost",
    "ppi_glass","ppi_plastic_bottles","ppi_steel",
    "deliv_hat_usd_per_l","z_base",
]

extra = [c for c in panel.columns if c not in cols_order]
panel_out = panel[cols_order + extra].sort_values(["week_start","country","market","grade"])

print("[INFO] Final weekly panel rows:", len(panel_out))
print("[INFO] week_start range:", panel_out["week_start"].min(), "→", panel_out["week_start"].max())

# ===================================================================
# WRITE OUTPUT
# ===================================================================
dest_prefix = f"s3://{bucket}/curated/weekly_panel/snapshot_date={snapshot}/"
dest = dest_prefix + "weekly_panel.parquet"

print("[INFO] Writing:", dest)
wr.s3.to_parquet(panel_out, dest, index=False)
print("[INFO] Weekly panel written successfully.")