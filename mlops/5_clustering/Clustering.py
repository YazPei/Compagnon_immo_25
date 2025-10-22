# path: mlops/5_clustering/Clustering.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings("ignore")

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

# --- Optional deps ---
try:
    import mlflow  # type: ignore
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False
    class _MLFlowNoOp:
        def set_tracking_uri(self, *a, **k): ...
        def set_experiment(self, *a, **k): ...
        def start_run(self, *a, **k): ...
        def end_run(self): ...
        def log_artifact(self, *a, **k): ...
    mlflow = _MLFlowNoOp()  # type: ignore

try:
    import geopandas as gpd  # type: ignore
    HAS_GPD = True
except Exception:
    HAS_GPD = False

# ---------------- helpers ----------------
def _diag():
    print("=== [clustering] diagnostics ===")
    print("python_exe:", sys.executable)
    print("python_ver:", sys.version.split()[0])
    print("cwd:", os.getcwd())
    print("sys_path_head:", sys.path[:5])
    print("HAS_MLFLOW:", HAS_MLFLOW)
    print("HAS_GPD:", HAS_GPD)
    print("MLFLOW_TRACKING_URI:", os.environ.get("MLFLOW_TRACKING_URI"))

def _setup_mlflow(exp_name: str = "Clustering Données Immo") -> None:
    if not HAS_MLFLOW:
        return
    uri = os.getenv("MLFLOW_TRACKING_URI")
    try:
        if uri:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(exp_name)
        else:
            raise RuntimeError("MLFLOW_TRACKING_URI non défini")
    except Exception:
        local_dir = Path("mlruns").resolve()
        local_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{local_dir}")
        mlflow.set_experiment(exp_name + " (offline)")

def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def _load_csv(file_path: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Introuvable: {file_path}")
    return pd.read_csv(file_path, sep=";", parse_dates=parse_dates, low_memory=False)

def _derive_code_postal(df: pd.DataFrame) -> pd.Series:
    if "codePostal" in df.columns:
        return df["codePostal"].astype(str).str.replace(r"\.0$", "", regex=True)
    return pd.Series(["inconnu"] * len(df), index=df.index)

def _code_postal_from_geo(df: pd.DataFrame, geo_file: Path) -> pd.Series:
    if not HAS_GPD or not geo_file.exists():
        return _derive_code_postal(df)
    need = {"mapCoordonneesLatitude", "mapCoordonneesLongitude"}
    if not need.issubset(df.columns):
        return _derive_code_postal(df)
    polys = gpd.read_file(geo_file)[["codePostal", "geometry"]].to_crs(epsg=4326)
    idx = df["mapCoordonneesLatitude"].notna() & df["mapCoordonneesLongitude"].notna()
    pts = gpd.GeoDataFrame(
        df.loc[idx].copy(),
        geometry=gpd.points_from_xy(df.loc[idx, "mapCoordonneesLongitude"], df.loc[idx, "mapCoordonneesLatitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, polys, how="left", predicate="within")
    cp = pd.Series("inconnu", index=df.index)
    cp.loc[idx] = (
        joined["codePostal"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .reindex(df.loc[idx].index)
        .fillna("inconnu")
    )
    return cp

def _plot_elbow(wcss: list[float], ks: list[int], out_png: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(ks, wcss, marker="o")
    plt.title("Elbow plot (KMeans)")
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()

def _cp_regroup(cp: str, frequents: set[str]) -> str:
    s = str(cp)
    if s in frequents:
        return s
    if s.startswith("97") and len(s) >= 3:
        return s[:3]
    if s.isdigit() and len(s) == 5:
        return s[:2]
    return "inconnu"

def _cp_final(zone: str) -> str:
    s = str(zone)
    if s.isdigit() and len(s) == 5:
        return s
    if s.isdigit() and len(s) == 2:
        return s + "000"
    if s.startswith("97") and len(s) == 3:
        return s + "00"
    return "inconnu"

# --------------- pipeline ----------------
def run_clustering_pipeline(input_path: str, output_path: str) -> None:
    _diag()
    _setup_mlflow()

    in_dir = Path(input_path)
    train_file = in_dir / "df_sales_clean_train.csv"
    test_file  = in_dir / "df_sales_clean_test.csv"
    geo_file   = in_dir / "contours-codes-postaux.geojson"  # optionnel

    # Vérifs entrées
    missing = [p for p in [train_file, test_file] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Fichiers manquants: {', '.join(map(str, missing))}")
    if HAS_GPD and not geo_file.exists():
        print(f"[INFO] GeoJSON absent ({geo_file}), fallback non-spatial.")

    out_path = Path(output_path)
    out_dir = out_path.parent if out_path.suffix.lower() == ".csv" else out_path
    out_cluster_csv = out_path if out_path.suffix.lower() == ".csv" else (out_dir / "df_cluster.csv")
    out_st_csv = out_dir / "df_sales_clean_ST.csv"

    _ensure_dirs(out_dir, Path("mlflow_outputs"), Path("exports"), Path("data"))

    ctx = mlflow.start_run(run_name="clustering_macro_kpi") if HAS_MLFLOW else _NullCtx()
    with ctx:
        # Load
        train = _load_csv(train_file, parse_dates=["date"])
        test  = _load_csv(test_file , parse_dates=["date"])
        print(f"[INFO] train shape: {train.shape}, test shape: {test.shape}")

        train["split"] = "train"
        test["split"]  = "test"
        df = pd.concat([train, test], ignore_index=True)

        # Code postal
        df["codePostal"] = _code_postal_from_geo(df, geo_file)

        # Zones mixtes
        cp_counts = df[df["split"] == "train"]["codePostal"].value_counts()
        cp_frequents = set(cp_counts[cp_counts >= 10].index)
        df["zone_mixte"] = df["codePostal"].astype(str).apply(lambda x: _cp_regroup(x, cp_frequents))

        # Agrégations mensuelles (train)
        tr = df[df["split"] == "train"].copy()
        tr = tr.dropna(subset=["date"])
        tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
        tr["Year"] = tr["date"].dt.year.astype(int)
        tr["Month"] = tr["date"].dt.month.astype(int)
        agg = (
            tr.groupby(["Year", "Month", "zone_mixte"])
              .agg(prix_m2_vente=("prix_m2_vente", "mean"))
              .reset_index()
        )
        if agg.empty:
            raise ValueError("Aucune agrégation produite (vérifie la colonne 'prix_m2_vente' et 'date').")

        agg["date"] = pd.to_datetime(dict(year=agg["Year"], month=agg["Month"], day=1))
        agg["codePostal_recons"] = agg["zone_mixte"].apply(_cp_final)
        agg = agg.sort_values(["codePostal_recons", "date"])
        agg["ym"] = agg["Year"] * 12 + agg["Month"]
        agg["t"] = agg.groupby("codePostal_recons")["ym"].transform(lambda x: x - x.min())
        agg["log_prix"] = np.log(agg["prix_m2_vente"])

        def tcam(g: pd.DataFrame) -> float:
            g = g.dropna(subset=["log_prix", "t"])
            if len(g) < 2:
                return np.nan
            coef = LinearRegression().fit(g[["t"]].values, g["log_prix"].values).coef_[0]
            return (np.exp(coef) - 1) * 100 * 12

        tcam_df = agg.groupby("codePostal_recons").apply(tcam).reset_index(name="tc_am_reg")

        cluster_input = (
            agg.rename(columns={"prix_m2_vente": "prix_m2_mean"})
               .groupby("codePostal_recons")
               .agg(prix_m2_mean=("prix_m2_mean", "mean"),
                    prix_m2_std =("prix_m2_mean", "std"),
                    prix_m2_max =("prix_m2_mean", "max"),
                    prix_m2_min =("prix_m2_mean", "min"))
               .reset_index()
        )
        cluster_input["prix_m2_cv"] = cluster_input["prix_m2_std"] / cluster_input["prix_m2_mean"]
        cluster_input = cluster_input.merge(tcam_df, on="codePostal_recons", how="left")
        if cluster_input.empty:
            raise ValueError("cluster_input vide — pas assez de données après agrégations.")

        cluster_input.to_csv("mlflow_outputs/cluster_input.csv", index=False, sep=";")
        if HAS_MLFLOW:
            mlflow.log_artifact("mlflow_outputs/cluster_input.csv")

        # Clustering
        feats = ["prix_m2_std", "prix_m2_max", "prix_m2_min", "tc_am_reg", "prix_m2_cv"]
        X = cluster_input[feats].replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty:
            raise ValueError("Features vides après nettoyage NaN/inf — impossible d’entraîner KMeans.")
        idx = X.index
        Xs = StandardScaler().fit_transform(X.values)

        ks = list(range(2, 9))
        wcss, sils, models = [], [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(Xs)
            models.append(km)
            wcss.append(km.inertia_)
            try:
                sils.append(silhouette_score(Xs, labels))
            except Exception:
                sils.append(np.nan)

        _plot_elbow(wcss, ks, Path("mlflow_outputs/elbow_plot.png"))
        if HAS_MLFLOW:
            mlflow.log_artifact("mlflow_outputs/elbow_plot.png")

        k_best = ks[int(np.nanargmax(sils))] if np.isfinite(sils).any() else 4
        km = models[ks.index(k_best)]
        labels = km.predict(Xs)
        cluster_input.loc[idx, "cluster"] = labels.astype(int)

        order = cluster_input.groupby("cluster")["prix_m2_mean"].mean().sort_values().index.tolist()
        names = ["Zones rurales/petites villes", "Centres urbains établis", "Banlieues mixtes", "Zones tendues/spéculatives"]
        name_map = dict(zip(order, names))
        cluster_input["cluster_label"] = cluster_input["cluster"].map(name_map)

        # Merge labels → full
        df["codePostal_recons"] = df["zone_mixte"].apply(_cp_final)
        lab = cluster_input[["codePostal_recons", "cluster", "cluster_label"]].drop_duplicates()
        out_full = df.merge(lab, on="codePostal_recons", how="left")
        out_full["cluster"] = out_full["cluster"].astype("Int64")
        out_full["cluster_label"] = out_full["cluster_label"].fillna("inconnu")

        # Exports
        out_full.drop(columns=["split"], errors="ignore").to_csv(out_st_csv, sep=";", index=False)
        out_full.to_csv(out_cluster_csv, sep=";", index=False)
        if HAS_MLFLOW:
            mlflow.log_artifact(str(out_st_csv))
            mlflow.log_artifact(str(out_cluster_csv))

        # Pour les stages suivants (copie standardisée)
        out_full.drop(columns=["split"], errors="ignore").to_csv(Path("data/df_sales_clean_ST.csv"), sep=";", index=False)
        out_full.to_csv(Path("data/df_cluster.csv"), sep=";", index=False)

        print(f"✅ Clustering OK → {out_cluster_csv}")
        print("ℹ️ geopandas :", "OK" if HAS_GPD else "non installé (fallback appliqué)")

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False

@click.command()
@click.option("--input-path", type=click.Path(exists=True, file_okay=False), required=True, help="Dossier source (data/processed)")
@click.option("--output-path", type=click.Path(), required=True, help="CSV de sortie (p.ex. exports/df_cluster.csv)")
def cli(input_path: str, output_path: str):
    try:
        run_clustering_pipeline(input_path, output_path)
    except Exception as e:
        print("[FATAL] clustering failed:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    cli()

