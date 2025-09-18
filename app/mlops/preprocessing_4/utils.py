import re
import os
import click
import pandas as pd
import polars as pl
import mlflow
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime
import math

run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")


# === Visualisation du dataset ===


# === Fonctions de nettoyage et utilitaires ===
def annee_const(x):
    bins = [
        float("-inf"),
        1948,
        1974,
        1977,
        1982,
        1988,
        2000,
        2005,
        2012,
        2021,
        float("inf"),
    ]
    labels = [
        "avant 1948",
        "1948-1974",
        "1975-1977",
        "1978-1982",
        "1983-1988",
        "1989-2000",
        "2001-2005",
        "2006-2012",
        "2013-2021",
        "après 2021",
    ]
    if "annee_construction" not in x.columns:
        raise ValueError("La colonne 'annee_construction' est manquante.")
    x["annee_construction_cat"] = pd.cut(
        x["annee_construction"], bins=bins, labels=labels, right=False
    )
    return x


# 1. Fonctions utilitaires


def clean_classe(x):
    """
    Nettoie et standardise les classes DPE/GES :
    - Vide, "Blank" ou "0" → np.nan
    - Conserve A→G, NS, VI
    - Sinon, extrait un code valide en début de chaîne via regex
    """
    # Cas manquant
    if pd.isna(x) or str(x).strip() in ["", "Blank", "0"]:
        return np.nan
    # Mise en majuscule et suppression des espaces
    s = str(x).strip().upper()
    # Acceptation stricte des codes connus
    if s in ["A", "B", "C", "D", "E", "F", "G", "NS", "VI"]:
        return s
    # Tentative de capture d'un code valide en début de chaîne
    m = re.match(r"^(NS|VI|[A-G])", s, re.IGNORECASE)
    return m.group(1).upper() if m else np.nan


def extract_principal(x):
    """
    Extrait la première source énergétique listée.
    Séparateurs gérés : ',', ';', '/', 'et'
    """
    # Cas manquant ou chaîne vide
    if pd.isna(x) or not str(x).strip():
        return np.nan
    # Split sur les séparateurs, on ne garde que la première partie
    parts = re.split(r"\s*(?:,|;|/|et)\s*", str(x).strip(), maxsplit=1)
    return parts[0] if parts else np.nan


# Configuration pour la normalisation de l'exposition
PATTERN_EXPO = r"(?i)\b(?:Nord(?:-Est|-Ouest)?|Sud(?:-Est|-Ouest)?|Est|Ouest|N|S|E|O)\b"
ORDRE_EXPO = ["Nord", "Est", "Sud", "Ouest"]
NORM_DIR = {
    "N": "Nord",
    "S": "Sud",
    "E": "Est",
    "O": "Ouest",
    "NORD": "Nord",
    "SUD": "Sud",
    "EST": "Est",
    "OUEST": "Ouest",
    "NORD-EST": "Nord-Est",
    "NORD-OUEST": "Nord-Ouest",
    "SUD-EST": "Sud-Est",
    "SUD-OUEST": "Sud-Ouest",
}


def clean_exposition(x):
    """
    Nettoie et standardise la colonne exposition :
    - Détecte les mots-clés de multi-exposition → "Multi-exposition"
    - Extrait les points cardinaux via regex
    - Traduit et normalise selon NORM_DIR
    - Trie et déduplique selon ORDRE_EXPO
    """
    # Cas manquant ou valeur vide
    if pd.isna(x) or str(x).strip() in ["", "0"]:
        return np.nan
    s = str(x).strip()
    low = s.lower()
    # 1) Multi-exposition via mots-clés
    if any(
        kw in low
        for kw in ["traversant", "multi", "toutes", "double expo", "triple", "360"]
    ):
        return "Multi-exposition"
    # 2) Extraction des directions
    matches = re.findall(PATTERN_EXPO, s, flags=re.IGNORECASE)
    dirs = [
        NORM_DIR[m.upper().replace(" ", "-")]
        for m in matches
        if m.upper().replace(" ", "-") in NORM_DIR
    ]
    # 3) Tri et déduplication
    uniq = sorted(set(dirs), key=lambda d: ORDRE_EXPO.index(d.split("-")[0]))
    return "-".join(uniq) if uniq else np.nan


######
# clean des variables numeriques
# Future colonne de regroupement pour les outliers
GROUP_COL = "INSEE_COM"  # colonne de regroupement


# Identification des colonnes numériques et exclusion des colonnes de coordonnées géographiques
def get_numeric_cols(x, group_col):
    """
    Retourne les colonnes numériques en excluant la future colonne de regroupement - qui est l'équivalent d'un code postal- pour les outliers à venir plus tard
    et les colonnes de coordonnées géographiques.
    """
    excluded_cols = ["mapCoordonneesLatitude", "mapCoordonneesLongitude"]
    return [
        col
        for col in x.select_dtypes(include="number").columns
        if col != group_col and col not in excluded_cols
    ]


def calculate_bounds(x, numeric_cols, lower_perc, upper_perc):
    """
    Calcule les bornes inférieures et supérieures pour chaque colonne numérique.
    """
    return {
        col: (x[col].quantile(lower_perc), x[col].quantile(upper_perc))
        for col in numeric_cols
    }


def compute_medians(train_data, bounds, group_col):
    """
    Calcule les médianes par groupe (group_col) et globales
    UNIQUEMENT à partir du train).
    """
    group_meds = {col: train_data.groupby(group_col)[col].median() for col in bounds}
    global_meds = train_data[list(bounds)].median()
    return group_meds, global_meds


OUTLIER_TAG = -999


def mark_outliers(x, bounds, outlier_tag=OUTLIER_TAG):
    """
    Pour chaque colonne dans bounds :
      - crée col+'_outlier_flag' = 1 si en dehors des bornes
      - remplace la valeur outlier par outlier_tag
    """
    for col, (low, high) in bounds.items():
        mask = (x[col] < low) | (x[col] > high)
        x[f"{col}_outlier_flag"] = mask.astype(int)
        x.loc[mask, col] = outlier_tag
    return x


def clean_outliers(
    df, bounds, group_meds, global_meds, group_col, outlier_tag=OUTLIER_TAG
):
    """
    Remplace dans df (train ou test) les tags outlier_tag par :
      - la médiane du groupe (via group_meds)
      - sinon la médiane globale (global_meds)
    """
    for col in bounds:
        mask = df[col] == outlier_tag
        # remplace uniquement les outliers taggés
        df.loc[mask, col] = (
            df.loc[mask, group_col]
            .map(group_meds[col])
            .fillna(global_meds[col])
            .astype(df[col].dtype)  # Ajout d'un cast pour éviter les problèmes de type
        )
    return df


def log_figure(fig, filename, artifact_path="figures"):
    """Sauvegarde une figure matplotlib et la loggue dans MLflow."""
    os.makedirs("outputs", exist_ok=True)
    full_path = os.path.join("outputs", filename)
    fig.savefig(full_path)
    mlflow.log_artifact(full_path, artifact_path=artifact_path)
    mlflow.set_tag("figure_logged", filename)

