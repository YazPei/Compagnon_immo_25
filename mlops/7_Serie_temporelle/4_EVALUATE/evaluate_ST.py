#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_ST.py — évaluation SARIMAX / Prophet / fallback
Version corrigée : support --mlflow-uri, --experiment, --run-id ; mlflow-safe (no-op si absent).
Usage:
  python evaluate_ST.py --input-folder data/split --model-folder outputs/best --output-folder outputs/evaluate --mlflow-uri "..." --experiment "ST-Eval"
"""
from __future__ import annotations
import os
import re
import glob
import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# optionally joblib/pickle for loading models
try:
    import joblib
except Exception:
    joblib = None
import pickle

# dotenv (si tu utilises .env.yaz)
try:
    from dotenv import load_dotenv
    if Path(".env.yaz").exists():
        load_dotenv(".env.yaz")
except Exception:
    pass

warnings.filterwarnings("ignore")


# ---------------- MLflow safe wrapper (no-op si absent) ----------------
class _NoOpRun:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

class _NoOpMLflow:
    def set_tracking_uri(self, *a, **k): pass
    def set_experiment(self, *a, **k): pass
    def start_run(self, *a, **k): return _NoOpRun()
    def active_run(self): return None
    def end_run(self, *a, **k): pass
    def log_metric(self, *a, **k): pass
    def log_param(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass
    def set_tag(self, *a, **k): pass

try:
    import mlflow as _mlflow
    MLFLOW = _mlflow
except Exception:
    MLFLOW = _NoOpMLflow()


# ---------------- helpers MLflow ----------------
def _mlflow_safe_set_tracking_uri(uri: Optional[str]) -> str:
    """Détermine et fixe URI (arg -> env -> fallback file:./mlruns) et retourne l'URI effective."""
    uri_eff = uri or os.getenv("MLFLOW_TRACKING_URI", None)
    if not uri_eff:
        uri_eff = f"file:{os.path.abspath('./mlruns')}"
    try:
        if hasattr(MLFLOW, "set_tracking_uri"):
            MLFLOW.set_tracking_uri(uri_eff)
    except Exception as e:
        print(f"[WARN] mlflow.set_tracking_uri failed: {e}")
    return uri_eff

def _mlflow_safe_set_experiment(name: str) -> None:
    try:
        if hasattr(MLFLOW, "set_experiment"):
            MLFLOW.set_experiment(name)
    except Exception as e:
        print(f"[WARN] mlflow.set_experiment failed: {e}")

def _mlflow_log_artifact(path: str, artifact_path: Optional[str] = None):
    try:
        if not os.path.exists(path):
            print(f"[WARN] artifact not found, skip: {path}")
            return
        if artifact_path:
            MLFLOW.log_artifact(path, artifact_path=artifact_path)
        else:
            MLFLOW.log_artifact(path)
    except Exception as e:
        print(f"[WARN] mlflow.log_artifact failed for {path}: {e}")

def _mlflow_log_metrics(metrics: dict):
    try:
        for k, v in metrics.items():
            MLFLOW.log_metric(k, float(v))
    except Exception as e:
        print(f"[WARN] mlflow.log_metric failed: {e}")

def _mlflow_log_params(params: dict):
    try:
        for k, v in params.items():
            MLFLOW.log_param(k, v)
    except Exception as e:
        print(f"[WARN] mlflow.log_param failed: {e}")


# ---------------- IO helpers ----------------
def resolve_split_paths(input_folder: str, suffix: Optional[str]) -> Tuple[str, str]:
    """
    Trouve automatiquement train/test (essaie variantes avec suffix puis q12 puis sans).
    """
    input_folder = str(Path(input_folder))
    suffix = suffix or ""
    candidates_train = [
        os.path.join(input_folder, f"train_periodique{suffix}.csv"),
        os.path.join(input_folder, "train_periodique_q12.csv"),
        os.path.join(input_folder, "train_periodique.csv"),
    ]
    candidates_test = [
        os.path.join(input_folder, f"test_periodique{suffix}.csv"),
        os.path.join(input_folder, "test_periodique_q12.csv"),
        os.path.join(input_folder, "test_periodique.csv"),
    ]
    train_path = next((p for p in candidates_train if os.path.exists(p)), None)
    test_path = next((p for p in candidates_test if os.path.exists(p)), None)
    if not train_path or not test_path:
        raise FileNotFoundError(
            "Impossible de trouver les splits.\n"
            f"Essayé (train): {candidates_train}\n"
            f"Essayé (test) : {candidates_test}"
        )
    return train_path, test_path

def infer_cluster_ids_from_models(model_folder: str) -> List[Optional[int]]:
    """
    Déduit les cluster_ids depuis les noms des fichiers modèles (regex 'cluster_(\\d+)').
    Si aucun modèle trouvé, retourne [None] — on fera une éval globale.
    """
    ids = set()
    for f in glob.glob(os.path.join(model_folder, "*.pkl")):
        m = re.search(r"cluster_(\d+)", os.path.basename(f))
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids) if ids else [None]

def pick_model_for_cluster(model_folder: str, cid: Optional[int]) -> Optional[str]:
    """
    Sélectionne un modèle pour le cluster cid. Si cid=None, prend le premier .pkl existant (si présent).
    """
    files = sorted(glob.glob(os.path.join(model_folder, "*.pkl")))
    if cid is None:
        return files[0] if files else None
    pattern = re.compile(rf"cluster_{cid}\b")
    for f in files:
        if pattern.search(os.path.basename(f)):
            return f
    return None

def load_model(model_path: str):
    """
    Charge un modèle .pkl via joblib si dispo, sinon pickle.
    """
    if model_path is None:
        return None
    if joblib is not None:
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ---------------- Forecast helpers ----------------
def forecast_with_model(model, test_index: pd.DatetimeIndex, steps: int, train_series: Optional[pd.Series] = None) -> np.ndarray:
    """
    Essaie get_forecast/predict, sinon fallback persistance (dernière valeur du train).
    Si model is None, utilise dernier point de train_series si fourni, sinon 0.0.
    """
    if model is None:
        # persistence from train_series
        if train_series is not None and len(train_series) > 0:
            last_val = float(pd.to_numeric(train_series.dropna()).iloc[-1])
        else:
            last_val = 0.0
        return np.full(steps, last_val, dtype=float)

    # statsmodels SARIMAXResults -> get_forecast
    if hasattr(model, "get_forecast"):
        try:
            fc = model.get_forecast(steps=steps)
            if hasattr(fc, "predicted_mean"):
                arr = np.asarray(fc.predicted_mean)
                if arr.shape[0] >= steps:
                    return arr[:steps]
                # else fallback to what we can
                return np.resize(arr, steps).astype(float)
        except Exception:
            pass

    # predict with start/end (many models accept that)
    if hasattr(model, "predict"):
        try:
            # try predict with pandas-like start/end
            try:
                yhat = model.predict(start=test_index[0], end=test_index[-1])
            except Exception:
                yhat = model.predict(steps=steps)
            arr = np.asarray(yhat, dtype=float)
            if arr.shape[0] >= steps:
                return arr[:steps]
            return np.resize(arr, steps).astype(float)
        except Exception:
            pass

    # fallback naive
    if train_series is not None and len(train_series) > 0:
        last_val = float(pd.to_numeric(train_series.dropna()).iloc[-1])
    else:
        # try to infer last from model attributes if possible
        last_val = 0.0
        for attr in ("endog", "data", "y"):
            obj = getattr(model, attr, None)
            if obj is None:
                continue
            try:
                arr = np.asarray(getattr(obj, "endog", getattr(obj, "y", obj)))
                if arr.size > 0:
                    last_val = float(arr[-1])
                    break
            except Exception:
                continue
    print("[WARN] fallback naïf (persistance) appliqué.")
    return np.full(steps, last_val, dtype=float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(np.asarray(y_true) == 0, 1e-8, np.asarray(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ---------------- Main evaluation flow ----------------
def evaluate(input_folder: str, output_folder: str, model_folder: str, suffix: Optional[str] = None,
             mlflow_uri: Optional[str] = None, experiment: str = "ST-SARIMAX-Evaluation", run_id: Optional[str] = None):
    # Setup mlflow (safe)
    effective_uri = _mlflow_safe_set_tracking_uri(mlflow_uri)
    _mlflow_safe_set_experiment(experiment)

    y_col = "prix_m2_vente"
    suffix = suffix or ""
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # find train/test
    train_path, test_path = resolve_split_paths(input_folder, suffix)
    print(f"[INFO] train={train_path}\n[INFO] test ={test_path}")

    df_train = pd.read_csv(train_path, sep=";", parse_dates=["date"]).set_index("date")
    df_test = pd.read_csv(test_path, sep=";", parse_dates=["date"]).set_index("date")

    # deduce cluster ids from available models
    cluster_ids = infer_cluster_ids_from_models(model_folder)
    if cluster_ids == [None] and "cluster" in df_train.columns:
        try:
            cluster_ids = sorted(df_train["cluster"].dropna().astype(int).unique().tolist())
        except Exception:
            cluster_ids = [None]

    # loop clusters
    for cid in (cluster_ids if cluster_ids != [None] else [None]):
        if cid is not None and "cluster" in df_train.columns:
            tr = df_train[df_train["cluster"] == cid]
            te = df_test[df_test["cluster"] == cid]
            if tr.empty or te.empty:
                print(f"[WARN] Pas de données pour cluster={cid}, on saute.")
                continue
            run_name = f"evaluate_cluster_{cid}{suffix}"
        else:
            tr, te = df_train, df_test
            run_name = f"evaluate_global{suffix}"

        if y_col not in tr.columns or y_col not in te.columns:
            raise KeyError(f"Colonne cible '{y_col}' absente du train/test.")

        model_path = pick_model_for_cluster(model_folder, cid)
        if model_path is None or not os.path.exists(model_path):
            print(f"[WARN] Aucun modèle trouvé pour cluster={cid}. On utilisera un fallback persistant.")
            model = None
        else:
            print(f"[INFO] Modèle utilisé (cluster={cid}): {model_path}")
            model = load_model(model_path)

        # forecast
        steps = len(te)
        y_pred = forecast_with_model(model, te.index, steps, train_series=tr[y_col] if y_col in tr.columns else None)
        y_pred = np.asarray(y_pred).reshape(-1)[:steps]
        y_true = np.asarray(te[y_col]).reshape(-1)

        metrics = compute_metrics(y_true, y_pred)

        # Save predictions
        pred_df = pd.DataFrame({"date": te.index, "y_true": y_true, "y_pred": y_pred})
        outfile = out_dir / (f"predictions_cluster_{cid}{suffix}.csv" if cid is not None else f"predictions_global{suffix}.csv")
        pred_df.to_csv(outfile, index=False, sep=";")

        # MLflow logging (safe)
        try:
            # if run_id provided, try to attach to it; otherwise create new run
            if run_id:
                run_ctx = MLFLOW.start_run(run_id=run_id)
            else:
                run_ctx = MLFLOW.start_run(run_name=run_name)
        except Exception:
            print("[WARN] mlflow.start_run failed — using no-op run context.")
            run_ctx = _NoOpRun()

        with run_ctx:
            try:
                if hasattr(MLFLOW, "log_param"):
                    MLFLOW.log_param("cluster_id", cid if cid is not None else "global")
                    MLFLOW.log_param("suffix", suffix)
                    if model_path:
                        MLFLOW.log_param("model_path", model_path)
                _mlflow_log_metrics(metrics)
                _mlflow_log_artifact(str(outfile), artifact_path="evaluate")
            except Exception as e:
                print(f"[WARN] erreur pendant mlflow logging: {e}")

        print(f"ok {run_name} -> {outfile} | {metrics}")

    print("OK - Evaluation terminée.")


# ---------------- CLI ----------------
def cli():
    parser = argparse.ArgumentParser(description="Evaluate SARIMAX / fallback models.")
    parser.add_argument("--input-folder", required=True, help="Folder containing train/test periodique CSVs (data/split)")
    parser.add_argument("--output-folder", required=True, help="Where to write predictions (outputs/evaluate)")
    parser.add_argument("--model-folder", required=True, help="Where models are stored (outputs/best)")
    parser.add_argument("--suffix", default="", help="Optional suffix used on split filenames")
    parser.add_argument("--mlflow-uri", default=None, help="Optional MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment", default="ST-SARIMAX-Evaluation", help="MLflow experiment name")
    parser.add_argument("--run-id", default=None, help="Attach to existing run id (optional)")
    args = parser.parse_args()
    evaluate(args.input_folder, args.output_folder, args.model_folder, suffix=args.suffix, mlflow_uri=args.mlflow_uri, experiment=args.experiment, run_id=args.run_id)

if __name__ == "__main__":
    cli()
