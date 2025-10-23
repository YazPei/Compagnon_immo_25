# path: mlops/preprocessing_4/utils.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def annee_const(df: pd.DataFrame) -> pd.DataFrame:
    """Catégorise 'annee_construction' si numérique; sinon no-op."""
    col = "annee_construction"
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        bins = [-1, 1947, 1974, 1977, 1982, 1988, 2000, 2100]
        labels = [
            "avant 1948",
            "1948-1974",
            "1975-1977",
            "1978-1982",
            "1983-1988",
            "1989-2000",
            "après 2000",
        ]
        try:
            df = df.copy()
            df[col] = pd.cut(s, bins=bins, labels=labels)
        except Exception:
            # why: ne pas bloquer si coupure impossible (données sales)
            pass
    return df


def clean_classe(x):
    """Normalise classes type DPE (A..G)."""
    if pd.isna(x):
        return x
    s = str(x).strip().upper()
    return s[:1] if s and s[0] in "ABCDEFG" else s


def clean_exposition(x):
    """Mappe directions FR vers codes compacts."""
    if pd.isna(x):
        return x
    s = str(x).lower()
    mapping = {
        "nord": "N",
        "sud": "S",
        "est": "E",
        "ouest": "O",
        "nord-est": "NE",
        "nord ouest": "NO",
        "nord-ouest": "NO",
        "sud-est": "SE",
        "sud-est": "SE",
        "sud-ouest": "SO",
    }
    for k, v in mapping.items():
        if k in s:
            return v
    return s


def extract_principal(x):
    """Retourne le premier item d'une liste séparée par virgule."""
    if pd.isna(x):
        return x
    s = str(x)
    return s.split(",")[0].strip()


def get_numeric_cols(df: pd.DataFrame, group_col: str | None = None) -> List[str]:
    """Retourne les colonnes numériques, excluant group_col si fourni."""
    cols = df.select_dtypes(
        include=["number", "float", "int", "Int64", "Float64"]
    ).columns.tolist()
    if group_col and group_col in cols:
        cols.remove(group_col)
    return cols


def calculate_bounds(
    df: pd.DataFrame, cols: List[str], low: float = 0.001, high: float = 0.999
) -> Dict[str, Tuple[float, float]]:
    """Bornes par quantiles [low, high] pour chaque colonne."""
    if not cols:
        return {}
    q = df[cols].quantile([low, high])
    bounds: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        if c not in df:
            continue
        lo = float(q.loc[low, c])
        hi = float(q.loc[high, c])
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            bounds[c] = (lo, hi)
    return bounds


def compute_medians(
    df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]], group_col: str, *_ , **__
):
    """Médianes par groupe et globale (numériques uniquement)."""
    grp = (
        df.groupby(group_col).median(numeric_only=True)
        if group_col in df.columns
        else pd.DataFrame()
    )
    glob = df.median(numeric_only=True)
    return grp, glob


def mark_outliers(
    df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """Ajoute <col>_outlier_flag=1 si en dehors [lo, hi]."""
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[f"{c}_outlier_flag"] = ((out[c] < lo) | (out[c] > hi)).astype(int)
    return out


def clean_outliers(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    group_medians: pd.DataFrame,
    global_medians: pd.Series,
    group_col: str,
) -> pd.DataFrame:
    """
    Remplace les outliers par la médiane globale (fallback si per-groupe indispo).
    """
    out = df.copy()
    for c in bounds:
        flag = f"{c}_outlier_flag"
        if c not in out.columns:
            continue
        # why: utiliser médiane globale si la médiane par groupe est indisponible
        med = (
            float(global_medians.get(c, np.nan))
            if pd.notna(global_medians.get(c, np.nan))
            else float(out[c].median())
        )
        mask = out.get(flag, 0) == 1
        out.loc[mask, c] = med
    return out

