#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import warnings
import math
import click
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
import joblib
import mlflow

warnings.filterwarnings("ignore", category=FutureWarning)

# ───── Colonnes / config ───────────────────────────────────────────────────────
TARGET = "prix_m2_vente"

ORDINAL_COLS = ["ges_class", "dpeL", "logement_neuf", "nb_pieces", "bain", "eau", "nb_toilettes", "balcon"]
ONEHOT_COLS  = ["typedebien", "typedetransaction", "chauffage_mode", "chauffage_energie_principal", "cluster"]
TARGET_COLS  = ["etage", "nb_etages", "exposition", "chauffage_energie", "chauffage_systeme", "date"]  # date → YYYY-MM
NUMERIC_COLS = [
    "surface", "surface_terrain", "dpeC", "places_parking", "charges_copro",
    "loyer_m2_median_n6", "nb_log_n6", "taux_rendement_n6",
    "loyer_m2_median_n7", "nb_log_n7", "taux_rendement_n7"
]
GEO_COLS     = ["x_geo", "y_geo", "z_geo"]
YEAR_COL     = ["annee_construction"]
YEAR_ORDER   = [
    "après 2021", "2013-2021", "2006-2012", "2001-2005",
    "1989-2000", "1983-1988", "1978-1982", "1975-1977",
    "1948-1974", "avant 1948"
]

# ───── Helpers géo & date ─────────────────────────────────────────────────────
def _radians(x):
    return x * math.pi / 180.0

def _infer_lat_lon(df: pd.DataFrame):
    candidates = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("lat", "lng"),
        ("Latitude", "Longitude"),
        ("LAT", "LON"),
    ]
    for la, lo in candidates:
        if la in df.columns and lo in df.columns:
            try:
                lat = pd.to_numeric(df[la], errors="coerce")
                lon = pd.to_numeric(df[lo], errors="coerce")
                return lat, lon
            except Exception:
                pass
    if "geometry" in df.columns:
        try:
            lat = df["geometry"].apply(lambda g: getattr(g, "y", np.nan)).astype(float)
            lon = df["geometry"].apply(lambda g: getattr(g, "x", np.nan)).astype(float)
            if lat.notna().any() and lon.notna().any():
                return lat, lon
        except Exception:
            pass
    return None, None

def add_geo_columns_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    need = [c for c in GEO_COLS if c not in df.columns]
    if not need:
        return df

    lat, lon = _infer_lat_lon(df)
    if lat is None or lon is None:
        print("⚠️  INFO: Impossible d'inférer les colonnes géo (lat/lon/geometry introuvables). "
              "Je poursuis sans x_geo/y_geo/z_geo.")
        return df

    lat = lat.clip(-90, 90)
    lon = lon.clip(-180, 180)
    lat_r = lat.apply(_radians)
    lon_r = lon.apply(_radians)

    x = (np.cos(lat_r) * np.cos(lon_r)).astype("float32")
    y = (np.cos(lat_r) * np.sin(lon_r)).astype("float32")
    z = (np.sin(lat_r)).astype("float32")

    df = df.copy()
    if "x_geo" not in df.columns: df["x_geo"] = x
    if "y_geo" not in df.columns: df["y_geo"] = y
    if "z_geo" not in df.columns: df["z_geo"] = z

    print("ℹ️  Colonnes géo générées: x_geo, y_geo, z_geo.")
    return df

def convert_date_for_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        try:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = df["date"].dt.to_period("M").astype(str)
        except Exception:
            pass
    return df

# ───── Year bucketizer (résout l’erreur initiale) ─────────────────────────────
_BINS = [-np.inf, 1947, 1974, 1977, 1982, 1988, 2000, 2005, 2012, 2021, np.inf]
_LABELS_ASC = ["avant 1948","1948-1974","1975-1977","1978-1982","1983-1988",
               "1989-2000","2001-2005","2006-2012","2013-2021","après 2021"]

def _year_bucketize_array(X):
    """Mappe années numériques ou libellés → libellés YEAR_ORDER (retour (n,1))."""
    s = pd.Series(np.ravel(X))
    s_str = s.astype(str)
    mask_in = s_str.isin(YEAR_ORDER)
    s_num = pd.to_numeric(s, errors="coerce")
    labels = pd.cut(
        s_num, bins=_BINS, labels=_LABELS_ASC,
        right=True, include_lowest=True
    ).astype(object)
    out = pd.Series(index=s.index, dtype=object)
    out[mask_in] = s_str[mask_in]
    out[~mask_in] = labels[~mask_in]
    out = out.fillna("1989-2000")
    return out.to_frame()

# ───── Fonctions top-level pour remplacer les lambdas (pickle-safe) ───────────
def _replace_neg1_with_5(X):
    X = np.asarray(X)
    return np.where(X == -1, 5, X)

def _plus1(X):
    X = np.asarray(X)
    return X + 1

def _invert_11_minus_X(X):
    X = np.asarray(X)
    return 11 - X

# ───── Pipelines ──────────────────────────────────────────────────────────────
def build_preprocessor_dynamic(X_fit: pd.DataFrame) -> ColumnTransformer:
    present_ord   = [c for c in ORDINAL_COLS if c in X_fit.columns]
    present_ohe   = [c for c in ONEHOT_COLS  if c in X_fit.columns]
    present_year  = [c for c in YEAR_COL     if c in X_fit.columns]
    present_tgt   = [c for c in TARGET_COLS  if c in X_fit.columns]
    present_num   = [c for c in NUMERIC_COLS if c in X_fit.columns]
    present_geo   = [c for c in GEO_COLS     if c in X_fit.columns]

    def warn_missing(expected, present, name):
        missing = [c for c in expected if c not in present]
        if missing:
            print(f"⚠️  {name}: colonnes manquantes ignorées: {missing}")

    warn_missing(ORDINAL_COLS, present_ord, "ORDINAL_COLS")
    warn_missing(ONEHOT_COLS,  present_ohe, "ONEHOT_COLS")
    warn_missing(YEAR_COL,     present_year,"YEAR_COL")
    warn_missing(TARGET_COLS,  present_tgt, "TARGET_COLS")
    warn_missing(NUMERIC_COLS, present_num, "NUMERIC_COLS")
    warn_missing(GEO_COLS,     present_geo, "GEO_COLS")

    ordinal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scale", StandardScaler())
    ])
    onehot_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    target_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("target", TargetEncoder())
    ])
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    geo_pipeline = Pipeline([
        ("scale", StandardScaler())
    ])

    # Year: bucketize -> ordinal (YEAR_ORDER) -> replace -1 -> +1 -> invert -> scale
    year_rank_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("bucket", FunctionTransformer(_year_bucketize_array, validate=False)),
        ("ord", OrdinalEncoder(categories=[YEAR_ORDER], dtype=int,
                               handle_unknown="use_encoded_value", unknown_value=-1)),
        ("replace", FunctionTransformer(_replace_neg1_with_5, validate=False)),
        ("plus1", FunctionTransformer(_plus1, validate=False)),
        ("invert", FunctionTransformer(_invert_11_minus_X, validate=False)),
        ("scale", StandardScaler())
    ])

    transformers = []
    if present_ord:  transformers.append(("ord",  ordinal_pipeline,   present_ord))
    if present_ohe:  transformers.append(("ohe",  onehot_pipeline,    present_ohe))
    if present_year: transformers.append(("year", year_rank_pipeline, present_year))
    if present_tgt:  transformers.append(("tar",  target_pipeline,    present_tgt))
    if present_num:  transformers.append(("num",  numeric_pipeline,   present_num))
    if present_geo:  transformers.append(("geo",  geo_pipeline,       present_geo))

    return ColumnTransformer(transformers, remainder="drop")

def get_feature_names(preprocessor: ColumnTransformer, X_fit: pd.DataFrame) -> list:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if not cols:
            continue
        cols_list = list(cols) if isinstance(cols, (list, tuple, np.ndarray)) else X_fit.columns[cols].tolist()
        if name == "ohe":
            ohe = trans.named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(cols_list).tolist())
        else:
            names.extend(cols_list)
    return names

# ───── CLI ────────────────────────────────────────────────────────────────────
@click.command()
@click.option("--data-path", default="data/df_cluster.csv", show_default=True, help="CSV brut d'entrée.")
@click.option("--output", default="data/encoded", show_default=True, help="Dossier de sortie des encodages.")
@click.option("--experiment", default="regression_pipeline", show_default=True, help="Nom d'expérience MLflow.")
@click.option("--random-state", default=42, show_default=True, help="Seed.")
def main(data_path, output, experiment, random_state):
    os.makedirs(output, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="encode_only"):
        print(f"Chargement: {data_path}")
        df = pd.read_csv(data_path, sep=';', low_memory=False)
        if TARGET not in df.columns:
            raise ValueError(f"Colonne cible absente: {TARGET}")

        # Cible et prétraitements légers
        df = df.dropna(subset=[TARGET]).copy()
        df = convert_date_for_target_encoding(df)
        df = add_geo_columns_if_missing(df)  # ← ajoute x_geo,y_geo,z_geo si possible

        # Split via colonne 'split' si présente, sinon 80/20
        if "split" in df.columns:
            train = df[df["split"] == "train"].copy()
            test  = df[df["split"] == "test"].copy()
            if train.empty or test.empty:
                print("⚠️  Split vide/partiel → bascule en split 80/20.")
                train, test = train_test_split(df, test_size=0.2, random_state=random_state)
        else:
            train, test = train_test_split(df, test_size=0.2, random_state=random_state)

        # Features disponibles dynamiquement
        all_candidate_cols = ORDINAL_COLS + ONEHOT_COLS + TARGET_COLS + NUMERIC_COLS + GEO_COLS + YEAR_COL
        present_cols = [c for c in all_candidate_cols if c in train.columns]

        X_train = train[present_cols].copy()
        y_train = train[TARGET].copy()
        X_test  = test[present_cols].copy()
        y_test  = test[TARGET].copy()

        preproc = build_preprocessor_dynamic(X_train)
        preproc.fit(X_train, y_train)  # TargetEncoder a besoin de y

        X_train_enc = preproc.transform(X_train)
        X_test_enc  = preproc.transform(X_test)
        encoded_columns = get_feature_names(preproc, X_train)

        # Sauvegarde
        paths = {
            "X_train": os.path.join(output, "X_train.csv"),
            "y_train": os.path.join(output, "y_train.csv"),
            "X_test":  os.path.join(output, "X_test.csv"),
            "y_test":  os.path.join(output, "y_test.csv"),
            "meta":    os.path.join(output, "encoding_metadata.json"),
            "preproc": os.path.join(output, "preprocessor.pkl"),
        }

        pd.DataFrame(X_train_enc, columns=encoded_columns).to_csv(paths["X_train"], sep=';', index=False)
        pd.DataFrame({TARGET: y_train}).to_csv(paths["y_train"], sep=';', index=False)
        pd.DataFrame(X_test_enc,  columns=encoded_columns).to_csv(paths["X_test"], sep=';', index=False)
        pd.DataFrame({TARGET: y_test}).to_csv(paths["y_test"], sep=';', index=False)

        joblib.dump(preproc, paths["preproc"])
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump({
                "encoded_columns": encoded_columns,
                "original_feature_columns_present": present_cols,
                "target": TARGET
            }, f, ensure_ascii=False, indent=2)

        # MLflow artifacts + params rapides
        for p in paths.values():
            mlflow.log_artifact(p)
        mlflow.log_param("n_features_encoded", int(len(encoded_columns)))
        mlflow.log_param("n_train_rows", int(len(y_train)))
        mlflow.log_param("n_test_rows", int(len(y_test)))

        print("✅ Encodage terminé. Fichiers écrits dans:", output)

if __name__ == "__main__":
    main()

