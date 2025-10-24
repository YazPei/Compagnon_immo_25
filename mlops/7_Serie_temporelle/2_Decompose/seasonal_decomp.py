#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seasonal_decomp.py — décomposition saisonnière par cluster, MLflow-safe & DVC-friendly.

Usage:
  python mlops/7_Serie_temporelle/2_Decompose/seasonal_decomp.py \
    --input-folder data/split \
    --output-folder outputs/decomposition \
    --suffix _v1 \
    --mlflow-uri "https://dagshub.com/xxx/yyy.mlflow" \
    --experiment "ST-Decomposition" \
    --expected-clusters 0 1 2 3
"""
from __future__ import annotations
import os
import argparse
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# MLflow optional
try:
    import mlflow
except Exception:
    mlflow = None

# ---------------- Helpers ----------------
def _ensure_dir(folder: str):
    Path(folder).mkdir(parents=True, exist_ok=True)

def _empty_csv(path: str):
    pd.DataFrame({"date": pd.to_datetime([], errors="coerce"), "value": []}).to_csv(path, sep=";", index=False)

def _mlflow_safe_set_tracking_uri(uri: Optional[str]):
    if mlflow is None:
        print("[WARN] mlflow non installé — skip tracking uri.")
        return None
    if not uri:
        uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if not uri:
        uri = f"file:{os.path.abspath('./mlruns')}"
    try:
        mlflow.set_tracking_uri(uri)
        print(f"[INFO] mlflow tracking uri set to: {uri}")
        return uri
    except Exception as e:
        print(f"[WARN] mlflow.set_tracking_uri failed: {e}")
        return uri

def _mlflow_safe_set_experiment(name: str):
    if mlflow is None:
        return
    try:
        mlflow.set_experiment(name)
    except Exception as e:
        print(f"[WARN] mlflow.set_experiment failed: {e}")

def _mlflow_log_artifact(path: str, artifact_path: Optional[str] = None):
    if mlflow is None:
        return
    if not os.path.exists(path):
        print(f"[WARN] artifact not exists, skip: {path}")
        return
    try:
        if artifact_path:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path)
    except Exception as e:
        print(f"[WARN] mlflow.log_artifact failed for {path}: {e}")

def _safe_decompose(y: pd.Series, model: str, period: int = 12):
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < max(24, 2 * period):
        raise ValueError(f"série trop courte pour une décomposition {model} (n={len(y)})")
    if float(y.var()) == 0.0:
        raise ValueError("variance nulle — impossible de décomposer")
    return seasonal_decompose(y, model=model, period=period)

def _save_fig_and_components(decomp, cluster_id, model, output_folder, suffix="") -> Dict[str, str]:
    _ensure_dir(output_folder)
    # figure principale
    fig = decomp.plot()
    fig.suptitle(f"Décomposition {model} - Cluster {cluster_id}")
    base_png = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}.png")
    plt.tight_layout()
    fig.savefig(base_png)
    plt.close(fig)

    if suffix:
        fig2 = decomp.plot()
        fig2.suptitle(f"Décomposition {model} - Cluster {cluster_id} ({suffix})")
        suff_png = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}{suffix}.png")
        plt.tight_layout()
        fig2.savefig(suff_png)
        plt.close(fig2)

    # components -> CSV
    def _to_df(series: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame({"date": series.index, "value": series.values})
        out["date"] = pd.to_datetime(out["date"])
        return out

    trend_df = _to_df(decomp.trend.dropna())
    seasonal_df = _to_df(decomp.seasonal.dropna())
    resid_df = _to_df(decomp.resid.dropna())

    trend_path    = os.path.join(output_folder, f"trend_{model}_cluster_{cluster_id}.csv")
    seasonal_path = os.path.join(output_folder, f"seasonal_{model}_cluster_{cluster_id}.csv")
    resid_path    = os.path.join(output_folder, f"resid_{model}_cluster_{cluster_id}.csv")

    trend_df.to_csv(trend_path, sep=";", index=False)
    seasonal_df.to_csv(seasonal_path, sep=";", index=False)
    resid_df.to_csv(resid_path, sep=";", index=False)

    if suffix:
        trend_df.to_csv(os.path.join(output_folder, f"trend_{model}_cluster_{cluster_id}{suffix}.csv"), sep=";", index=False)
        seasonal_df.to_csv(os.path.join(output_folder, f"seasonal_{model}_cluster_{cluster_id}{suffix}.csv"), sep=";", index=False)
        resid_df.to_csv(os.path.join(output_folder, f"resid_{model}_cluster_{cluster_id}{suffix}.csv"), sep=";", index=False)

    return {"png": base_png, "trend": trend_path, "seasonal": seasonal_path, "resid": resid_path}

def _write_placeholders(output_folder: str, cluster_id, model: str, suffix: str, reason: str) -> Dict[str, str]:
    """Crée fichiers placeholder (PNG + CSVs) pour DVC si décomposition impossible."""
    _ensure_dir(output_folder)
    png_path = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}.png")
    plt.figure(figsize=(8, 3))
    plt.text(0.5, 0.5, f"No decomposition for cluster {cluster_id}\n({model})\nReason: {reason}",
             ha="center", va="center", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    trend_path    = os.path.join(output_folder, f"trend_{model}_cluster_{cluster_id}.csv")
    seasonal_path = os.path.join(output_folder, f"seasonal_{model}_cluster_{cluster_id}.csv")
    resid_path    = os.path.join(output_folder, f"resid_{model}_cluster_{cluster_id}.csv")
    _empty_csv(trend_path); _empty_csv(seasonal_path); _empty_csv(resid_path)

    if suffix:
        png_suff = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}{suffix}.png")
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, f"No decomposition for cluster {cluster_id}\n({model}, {suffix})\nReason: {reason}",
                 ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png_suff)
        plt.close()
        for base, p in [("trend", trend_path), ("seasonal", seasonal_path), ("resid", resid_path)]:
            pd.read_csv(p, sep=";").to_csv(
                os.path.join(output_folder, f"{base}_{model}_cluster_{cluster_id}{suffix}.csv"),
                sep=";", index=False
            )
    return {"png": png_path, "trend": trend_path, "seasonal": seasonal_path, "resid": resid_path}

# --------- Chargement données (avec fallbacks) ---------
def _load_periodique_concat(input_folder: str) -> Optional[pd.DataFrame]:
    """Charge train/test_periodique_q12*.csv si dispo, sinon None."""
    if not os.path.isdir(input_folder):
        return None
    candidates = []
    for base in ["train_periodique_q12", "test_periodique_q12"]:
        for f in sorted(os.listdir(input_folder)):
            if f.startswith(base) and f.endswith(".csv"):
                candidates.append(os.path.join(input_folder, f))
    if not candidates:
        return None
    dfs = []
    for path in candidates:
        try:
            df = pd.read_csv(path, sep=";", parse_dates=["date"])
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] lecture échouée {path}: {e}")
    if not dfs:
        return None
    full = pd.concat(dfs, ignore_index=True)
    required = {"date", "cluster", "prix_m2_vente"}
    missing = required - set(full.columns)
    if missing:
        print(f"[WARN] colonnes manquantes dans periodique_q12: {missing}")
        return None
    full = full.sort_values(["cluster", "date"]).reset_index(drop=True)
    return full

def _load_from_st_fallback(repo_root: str = ".") -> Optional[pd.DataFrame]:
    """Fallback depuis exports/df_sales_clean_ST.csv (on reconstruit séries mensuelles)."""
    st_path = os.path.join(repo_root, "exports", "df_sales_clean_ST.csv")
    if not os.path.exists(st_path):
        return None
    try:
        df = pd.read_csv(st_path, sep=";", low_memory=False)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "cluster" not in df.columns:
            df["cluster"] = 0  # cluster unique si pas de colonne
        if "prix_m2_vente" not in df.columns:
            # dernier recours
            if "prix_bien" in df.columns and "surface" in df.columns:
                df["prix_m2_vente"] = pd.to_numeric(df["prix_bien"], errors="coerce") / pd.to_numeric(df["surface"], errors="coerce")
            else:
                return None
        df = df.dropna(subset=["date", "prix_m2_vente"])
        # agrégations mensuelles par cluster
        df["ym"] = df["date"].dt.to_period("M").astype(str)
        agg = (df.groupby(["cluster","ym"])["prix_m2_vente"].mean().reset_index())
        agg["date"] = pd.to_datetime(agg["ym"] + "-01")
        full = agg[["date","cluster","prix_m2_vente"]].sort_values(["cluster","date"]).reset_index(drop=True)
        return full
    except Exception as e:
        print(f"[WARN] fallback ST échoué: {e}")
        return None

# ---------------- Core ----------------
def run_decomposition(input_folder: str,
                      output_folder: str,
                      suffix: str = "",
                      expected_clusters: Optional[List[int]] = None,
                      mlflow_uri: Optional[str] = None,
                      experiment: str = "ST-Decomposition"):
    _ensure_dir(output_folder)
    effective_uri = _mlflow_safe_set_tracking_uri(mlflow_uri)
    _mlflow_safe_set_experiment(experiment)

    # 1) periodique_q12 si dispo, sinon fallback ST
    full = _load_periodique_concat(input_folder)
    if full is None:
        print("[WARN] Aucun *periodique_q12*.csv — tentative fallback via exports/df_sales_clean_ST.csv")
        full = _load_from_st_fallback(".")
    if full is None or full.empty:
        print("[WARN] Aucune donnée exploitable (periodique_q12 ni ST). Génération de placeholders.")

    data_clusters = sorted([int(c) for c in full["cluster"].dropna().unique()]) if full is not None and not full.empty else []
    if not expected_clusters:
        expected_clusters = data_clusters if data_clusters else [0,1,2,3]

    for cluster_id in expected_clusters:
        if full is None or full.empty:
            dfc = pd.DataFrame(columns=["date","prix_m2_vente"]).set_index(pd.to_datetime([]))
        else:
            dfc = full[full["cluster"] == cluster_id].copy()
            dfc = dfc.sort_values("date").set_index("date")

        y = dfc["prix_m2_vente"] if "prix_m2_vente" in dfc.columns else pd.Series(dtype=float)

        # MLflow run
        if mlflow is not None:
            try:
                run_ctx = mlflow.start_run(run_name=f"decomp_cluster_{cluster_id}{suffix}")
            except Exception as e:
                print(f"[WARN] mlflow.start_run failed: {e}; continuing without run context.")
                run_ctx = None
        else:
            run_ctx = None
        if run_ctx is not None:
            run_ctx.__enter__()

        try:
            for model in ["additive","multiplicative"]:
                try:
                    if dfc.empty or y.dropna().empty:
                        raise ValueError("aucune donnée pour ce cluster")
                    decomp = _safe_decompose(y, model=model, period=12)
                    paths = _save_fig_and_components(decomp, cluster_id, model, output_folder, suffix=suffix)

                    # Log artifacts + metrics
                    _mlflow_log_artifact(paths["png"])
                    _mlflow_log_artifact(paths["trend"])
                    _mlflow_log_artifact(paths["seasonal"])
                    _mlflow_log_artifact(paths["resid"])
                    if mlflow is not None:
                        try:
                            mlflow.log_param("cluster_id", int(cluster_id))
                            mlflow.log_param("model", model)
                            mlflow.log_metric(f"{model}_trend_mean", float(decomp.trend.dropna().mean()))
                            mlflow.log_metric(f"{model}_resid_std", float(decomp.resid.dropna().std()))
                            mlflow.log_metric(f"{model}_resid_skew", float(decomp.resid.dropna().skew()))
                        except Exception as e:
                            print(f"[WARN] mlflow.log_metric failed: {e}")
                except Exception as e:
                    # placeholders + tags
                    paths = _write_placeholders(output_folder, cluster_id, model, suffix, reason=str(e))
                    if mlflow is not None:
                        try:
                            mlflow.set_tag(f"{model}_placeholder", True)
                            mlflow.set_tag(f"{model}_reason", str(e))
                            _mlflow_log_artifact(paths["png"])
                            _mlflow_log_artifact(paths["trend"])
                            _mlflow_log_artifact(paths["seasonal"])
                            _mlflow_log_artifact(paths["resid"])
                        except Exception as ex:
                            print(f"[WARN] mlflow logging for placeholder failed: {ex}")
        finally:
            if run_ctx is not None:
                try:
                    run_ctx.__exit__(None, None, None)
                except Exception:
                    pass

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Décomposition saisonnière par cluster.")
    parser.add_argument("--input-folder", type=str, required=True, help="Dossier des séries (data/split)")
    parser.add_argument("--output-folder", type=str, required=True, help="Dossier de sortie (ex: outputs/decomposition)")
    parser.add_argument("--suffix", type=str, default="", help="Suffixe optionnel (créera des copies)")
    parser.add_argument("--expected-clusters", type=int, nargs="*", default=None, help="Clusters à forcer (ex: --expected-clusters 0 1 2 3)")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="(optionnel) MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment", type=str, default="ST-Decomposition", help="Nom de l'expérience MLflow")
    args = parser.parse_args()
    run_decomposition(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        suffix=args.suffix,
        expected_clusters=args.expected_clusters,
        mlflow_uri=args.mlflow_uri,
        experiment=args.experiment,
    )

