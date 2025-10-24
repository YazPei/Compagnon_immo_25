#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, Tuple, Iterable
import numpy as np
import pandas as pd

_YEAR_BINS = [-np.inf, 1947, 1974, 1977, 1982, 1988, 2000, 2005, 2012, 2021, np.inf]
_YEAR_LABS = ["avant_1948", "1948_1974", "1975_1977", "1978_1982", "1983_1988",
              "1989_2000", "2001_2005", "2006_2012", "2013_2021", "apres_2021"]

def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")

def annee_const(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    src_candidates = ["annee_construction", "anneeConstruction", "annee_const", "year_built", "ANNEE_CONS"]
    src = next((c for c in src_candidates if c in df.columns), None)
    if src is None:
        df["annee_construction"] = pd.Categorical(["1989_2000"] * len(df), categories=_YEAR_LABS, ordered=True)
        return df

    col = df[src]
    if pd.api.types.is_string_dtype(col) or col.dtype == object:
        ok_mask = col.astype(str).isin(_YEAR_LABS)
        out = pd.Series(index=col.index, dtype="object")
        out[ok_mask] = col[ok_mask].astype(str)
        num = _to_numeric(col.where(~ok_mask))
        buck = pd.cut(num, bins=_YEAR_BINS, labels=_YEAR_LABS)
        out[~ok_mask] = buck.astype(str)
        out = out.fillna("1989_2000")
        df["annee_construction"] = pd.Categorical(out, categories=_YEAR_LABS, ordered=True)
        return df

    num = _to_numeric(col)
    buck = pd.cut(num, bins=_YEAR_BINS, labels=_YEAR_LABS)
    df["annee_construction"] = pd.Categorical(buck.astype(str).fillna("1989_2000"),
                                              categories=_YEAR_LABS, ordered=True)
    return df

def clean_classe(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NP"
    s = str(x).strip().upper()
    m = re.search(r"[A-G]", s)
    return m.group(0) if m else "NP"

def clean_exposition(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    s = str(x).lower()
    s = s.replace("ouest", "o").replace("est", "e").replace("sud", "s").replace("nord", "n")
    s = s.replace("west", "o").replace("east", "e").replace("south", "s").replace("north", "n")
    dirs = ["ne", "se", "so", "no", "n", "e", "s", "o"]
    for d in dirs:
        if re.search(rf"\b{d}\b", s):
            return d.upper()
    letters = "".join(sorted(set([c for c in s if c in "neso"]), key=lambda c: "neso".index(c)))
    return letters.upper() if letters else "NA"

def extract_principal(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "inconnu"
    s = re.split(r"[+,/;|-]+", str(x))
    s = [t.strip() for t in s if t.strip()]
    return (s[0] if s else "inconnu").lower()

def get_numeric_cols(df: pd.DataFrame, group_col: str | None = None) -> list:
    nums = []
    for c in df.columns:
        try:
            s = pd.to_numeric(df[c].dropna().head(50), errors="coerce")
            if s.notna().any():
                nums.append(c)
        except Exception:
            continue
    if group_col and group_col in nums:
        nums.remove(group_col)
    return nums

def calculate_bounds(df: pd.DataFrame, cols: Iterable[str], q_low: float = 0.001, q_high: float = 0.999) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = _to_numeric(df[c])
        s_nonull = s.dropna()
        if s_nonull.empty:
            continue
        try:
            lo = float(s_nonull.quantile(q_low))
            hi = float(s_nonull.quantile(q_high))
        except Exception:
            q1 = float(s_nonull.quantile(0.25))
            q3 = float(s_nonull.quantile(0.75))
            iqr = q3 - q1
            lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            med = float(s_nonull.median(skipna=True))
            if np.isfinite(med):
                lo, hi = med - 1.0, med + 1.0
            else:
                continue
        bounds[c] = (lo, hi)
    return bounds

def compute_medians(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]], group_col: str) -> tuple[Dict[str, pd.Series], Dict[str, float]]:
    group_meds: Dict[str, pd.Series] = {}
    global_meds: Dict[str, float] = {}
    if not bounds:
        return group_meds, global_meds
    g = df.groupby(group_col, dropna=False) if group_col in df.columns else None
    for c in bounds.keys():
        s = _to_numeric(df[c])
        if s.dropna().empty:
            global_meds[c] = float("nan")
            continue
        global_meds[c] = float(s.median(skipna=True))
        if g is not None:
            try:
                group_meds[c] = g[c].median()
            except Exception:
                group_meds[c] = pd.Series(dtype="float64")
    return group_meds, global_meds

def mark_outliers(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    if not bounds:
        return df.copy()
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        if c not in out.columns:
            continue
        s = _to_numeric(out[c])
        flag = ((s < lo) | (s > hi)).astype(int)
        out[f"{c}_outlier_flag"] = flag
    return out

def clean_outliers(df: pd.DataFrame,
                   bounds: Dict[str, Tuple[float, float]],
                   group_medians: Dict[str, pd.Series],
                   global_medians: Dict[str, float],
                   group_col: str) -> pd.DataFrame:
    if not bounds:
        return df.copy()
    out = df.copy()
    has_group = group_col in out.columns and bool(group_medians)
    for c, (lo, hi) in bounds.items():
        if c not in out.columns:
            continue
        flag_col = f"{c}_outlier_flag"
        if flag_col not in out.columns:
            continue
        idx = out[flag_col] == 1
        if not idx.any():
            continue
        if has_group and c in group_medians:
            med_by_g = group_medians[c]
            def impute_row(r):
                g = r.get(group_col, None)
                val = med_by_g.get(g, np.nan) if g in med_by_g.index else np.nan
                return val if np.isfinite(val) else global_medians.get(c, np.nan)
            out.loc[idx, c] = out.loc[idx].apply(impute_row, axis=1)
        else:
            out.loc[idx, c] = global_medians.get(c, np.nan)
    return out
