#!/usr/bin/env python3
"""
Robust helper: build data/split/train_periodique_q12.csv from exports/df_sales_clean_ST.csv
Heuristiques:
 - cherche une colonne date (date, periode, month, periode_mois, etc.) et une colonne value (prix_m2_vente, value, target, prix)
 - si pas de cluster, crée cluster=0
 - agrège en moyenne par periode mensuelle (index date -> YYYY-MM-01)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path("exports/df_sales_clean_ST.csv")
OUTDIR = Path("data/split")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "train_periodique_q12.csv"

if not SRC.exists():
    print("SOURCE missing:", SRC, file=sys.stderr)
    sys.exit(2)

print("Reading", SRC)
df = pd.read_csv(SRC, sep=",", low_memory=False)

# heuristiques colonnes date
date_candidates = ["date","period","periode","periode_date","month","mois","date_obs","timestamp"]
date_col = None
for c in date_candidates:
    if c in df.columns:
        # try parse
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        if parsed.notna().sum() > 0:
            date_col = c
            df[ "_parsed_date" ] = parsed
            break

# fallback: try any object column that parses
if date_col is None:
    for c in df.columns:
        if df[c].dtype == object:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > 0:
                date_col = c
                df[ "_parsed_date" ] = parsed
                break

if date_col is None:
    print("No parsable date column found. Inspect exports/df_sales_clean_ST.csv manually.", file=sys.stderr)
    sys.exit(3)

# find value column
value_candidates = ["prix_m2_vente","prix_m2","prix","value","target","y"]
value_col = next((c for c in value_candidates if c in df.columns), None)
if value_col is None:
    # pick first numeric-like column that is not the date
    for c in df.columns:
        if c == date_col: continue
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().sum() > 0:
            value_col = c
            break

if value_col is None:
    print("No numeric value column found. Cannot build series.", file=sys.stderr)
    sys.exit(4)

# cluster col?
cluster_candidates = ["cluster","cluster_id","kmeans","cluster_label"]
cluster_col = next((c for c in cluster_candidates if c in df.columns), None)
if cluster_col is None:
    df["_cluster"] = 0
    cluster_col = "_cluster"

# create monthly period
df["_date_mon"] = pd.to_datetime(df["_parsed_date"]).dt.to_period("M").dt.to_timestamp()
# aggregate mean by month & cluster
agg = df.groupby(["_date_mon", cluster_col], dropna=False)[value_col].mean().reset_index()
agg = agg.rename(columns={"_date_mon":"date","_cluster": "cluster", value_col:"value"})
# ensure date format ISO
agg["date"] = pd.to_datetime(agg["date"]).dt.strftime("%Y-%m-%d")

# write
agg.to_csv(OUT, index=False)
print("Wrote", OUT, "rows:", len(agg))
