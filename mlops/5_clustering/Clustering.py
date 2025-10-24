#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, traceback
from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Configure MLflow (prefer env var if present)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/yazpei/Compagnon_immo_25.mlflow")
mlflow.set_tracking_uri(MLFLOW_URI)

def log(*args):
    msg = " ".join(str(a) for a in args)
    enc = sys.stdout.encoding or "utf-8"
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))

def ensure_dirs(*paths: Path) -> None:
    for p in paths: p.mkdir(parents=True, exist_ok=True)

def load_csv(p: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    if not p.exists(): raise FileNotFoundError(f"Introuvable: {p}")
    return pd.read_csv(p, sep=";", parse_dates=parse_dates, low_memory=False)

def cp_str(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.replace(r"\.0$", "", regex=True)
    return out.where(out.str.match(r"^\d{5}$"), other="inconnu")

def departement_from_cp(cp: pd.Series) -> pd.Series:
    cp = cp_str(cp)
    dep = cp.str[:2].where(cp != "inconnu", other="inconnu")
    dep = dep.mask(cp.str.startswith("97"), cp.str[:3])
    dep = dep.mask(cp.str.startswith("20"), "2A")
    return dep.fillna("inconnu")

def month_key(dt: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt, errors="coerce")
    return (dt.dt.year.astype("Int64").astype(str) + "-" +
            dt.dt.month.astype("Int64").astype(str).str.zfill(2))

def tcam_from_series(df: pd.DataFrame, y_col: str = "prix_m2_vente") -> pd.Series:
    required = {"zone", "t", y_col}
    if not required.issubset(df.columns):
        zones = df["zone"].unique() if "zone" in df.columns else []
        return pd.Series({z: np.nan for z in zones})

    g = df.dropna(subset=[y_col, "t"]).copy()
    if g.empty:
        zones = df["zone"].unique()
        return pd.Series({z: np.nan for z in zones})

    g["logy"] = np.log(g[y_col].astype(float).clip(lower=1e-9))

    def _one(sub: pd.DataFrame) -> float:
        sub = sub.dropna(subset=["logy", "t"])
        if len(sub) < 2:
            return np.nan
        coef = LinearRegression().fit(sub[["t"]].values, sub["logy"].values).coef_[0]
        return (np.exp(coef) - 1) * 100 * 12

    return g.groupby("zone").apply(_one)

def ensure_clusters_always(df_full: pd.DataFrame, zone_col: str, k_default: int = 4) -> pd.DataFrame:
    """Retourne un DF avec colonnes cluster (int) et cluster_label (str)."""
    # aggregate by zone & ym
    agg = (df_full
           .groupby(["ym", zone_col], dropna=False)
           .agg(prix_m2_vente=("prix_m2_vente", "mean"))
           .reset_index())
    if agg.empty:
        log("WARN: Aucune aggregation produite, fallback quantiles globaux.")
        quant = pd.qcut(df_full["prix_m2_vente"], q=4, labels=False, duplicates="drop")
        labels = quant.fillna(0).astype(int)
        df_full["cluster"] = labels
        name_map = {0: "bas", 1: "moyen-", 2: "moyen+", 3: "haut"}
        df_full["cluster_label"] = df_full["cluster"].map(name_map).fillna("moyen")
        return df_full

    ym = pd.to_datetime(agg["ym"] + "-01", errors="coerce")
    agg["ym_num"] = ym.dt.year * 12 + ym.dt.month
    agg["t"] = agg.groupby(zone_col)["ym_num"].transform(lambda x: x - x.min())
    agg["zone"] = agg[zone_col].astype(str)

    tcam = tcam_from_series(agg, y_col="prix_m2_vente")

    feats = (agg.groupby(zone_col)
               .agg(y_mean=("prix_m2_vente","mean"),
                    y_std =("prix_m2_vente","std"),
                    y_min =("prix_m2_vente","min"),
                    y_max =("prix_m2_vente","max"))
               .reset_index())
    feats["cv"] = feats["y_std"] / feats["y_mean"]
    feats["tcam"] = feats[zone_col].map(tcam)

    X = feats[["y_mean","y_std","y_min","y_max","cv","tcam"]].replace([np.inf,-np.inf], np.nan)
    Xn = X.dropna()
    if len(Xn) < 2:
        log("WARN: Trop peu de zones pour KMeans, fallback quantiles y_mean.")
        q = pd.qcut(feats["y_mean"], q=min(4, max(1, len(feats))), labels=False, duplicates="drop")
        feats["cluster"] = q.fillna(0).astype(int)
    else:
        k_max = min(k_default, len(Xn))
        if k_max < 2: k_max = 2
        Xs = StandardScaler().fit_transform(Xn.values)

        ks = list(range(2, min(8, len(Xn))+1))
        models, sils = [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            lab = km.fit_predict(Xs)
            models.append(km)
            try:
                sils.append(silhouette_score(Xs, lab))
            except Exception:
                sils.append(np.nan)
            mlflow.set_experiment("Clustering")
            with mlflow.start_run(run_name="clustering"):
                mlflow.log_param("n_zones", int(n_zones))
                mlflow.log_param("final_n_clusters", int(final_k))
                mlflow.log_metric("silhouette", float(silhouette) if not np.isnan(silhouette) else float('nan'))
        if any(np.isfinite(sils)):
            k_best = ks[int(np.nanargmax(sils))]
        else:
            k_best = min(4, len(Xn))
        if k_best >= len(Xn):
            k_best = max(2, len(Xn)-1)
        km = models[ks.index(k_best)]
        labels = pd.Series(km.predict(Xs), index=Xn.index)
        feats["cluster"] = 0
        feats.loc[Xn.index, "cluster"] = labels.astype(int)

    # readable labels
    order = feats.groupby("cluster")["y_mean"].mean().sort_values().index.tolist()
    names = ["bas", "moyen-", "moyen+", "haut", "tres haut", "premium"]
    name_map = {c: names[i % len(names)] for i, c in enumerate(order)}
    feats["cluster_label"] = feats["cluster"].map(name_map)

    # merge back to full df
    out = df_full.merge(feats[[zone_col, "cluster", "cluster_label"]],
                        on=zone_col, how="left")
    out["cluster"] = out["cluster"].fillna(0).astype(int)
    out["cluster_label"] = out["cluster_label"].fillna("moyen")
    return out

def fallback_quantiles(df_full: pd.DataFrame, zone_col: str) -> pd.DataFrame:
    q = pd.qcut(df_full["prix_m2_vente"], q=4, labels=False, duplicates="drop")
    df_full = df_full.copy()
    df_full["cluster"] = q.fillna(0).astype(int)
    name_map = {0: "bas", 1: "moyen-", 2: "moyen+", 3: "haut"}
    df_full["cluster_label"] = df_full["cluster"].map(name_map).fillna("moyen")
    return df_full

@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="Dossier contenant df_sales_clean_train.csv et df_sales_clean_test.csv")
@click.option("--output-dir", type=click.Path(), default="data", show_default=True,
              help="Dossier sortie (écrit df_cluster.csv et df_sales_clean_ST.csv)")
def main(input_dir: str, output_dir: str):
    try:
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        ensure_dirs(out_dir)

        train = load_csv(in_dir / "df_sales_clean_train.csv", parse_dates=["date"])
        test  = load_csv(in_dir / "df_sales_clean_test.csv",  parse_dates=["date"])
        train["split"] = "train"; test["split"] = "test"
        df = pd.concat([train, test], ignore_index=True)

        if "codePostal" in df.columns:
            df["codePostal"] = cp_str(df["codePostal"])
        else:
            df["codePostal"] = "inconnu"
        df["dep_zone"] = departement_from_cp(df["codePostal"])
        df["ym"] = month_key(df["date"])

        if "prix_m2_vente" not in df.columns:
            raise ValueError("Colonne prix_m2_vente absente.")
        df = df.dropna(subset=["prix_m2_vente"]).copy()

        # Start an MLflow run for the clustering stage
        mlflow.set_experiment("Clustering")
        with mlflow.start_run(run_name="clustering_run"):
            log("MLflow run started:", mlflow.get_tracking_uri())
            SAFE = os.getenv("CLUSTERING_SAFE", "0") == "1"
            try:
                out = ensure_clusters_always(df, zone_col="dep_zone", k_default=4)
                # log some basic metrics
                try:
                    mlflow.log_param("n_rows", int(len(df)))
                    mlflow.log_param("n_zones", int(df["dep_zone"].nunique()))
                except Exception:
                    pass
            except Exception as e:
                if SAFE:
                    log(f"[WARN] Pipeline clustering en echec ({e}), fallback quantiles.")
                    out = fallback_quantiles(df, "dep_zone")
                else:
                    raise

            st_path = out_dir / "df_sales_clean_ST.csv"
            cl_path = out_dir / "df_cluster.csv"
            out.drop(columns=["split"], errors="ignore").to_csv(st_path, sep=";", index=False)
            out.to_csv(cl_path, sep=";", index=False)

            try:
                mlflow.log_artifact(str(cl_path), artifact_path="clustering_output")
                mlflow.log_artifact(str(st_path), artifact_path="clustering_output")
            except Exception:
                pass

            log("OK: ecrit", cl_path, "et", st_path)
    except Exception as e:
        log("[FATAL] clustering failed:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

