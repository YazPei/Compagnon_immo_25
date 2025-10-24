#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyse.py — charge modèle, calcule métriques, génère visuels (SHAP) et log dans MLflow.
Usage: python mlops/6_Regression/4_Analyse/analyse.py --help
"""
from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path

import click
import joblib
import pandas as pd
import mlflow

# Utils (métriques/plots)
UTILS_DIR = Path(__file__).resolve().parent.parent / "3_UTILS"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from utils import compute_metrics, print_metrics, shap_summary_plot  # noqa: E402

# ---------------- helper mlflow ----------------
def _mlflow_safe_set_tracking_uri(uri: str | None) -> str:
    """
    Détermine et fixe le tracking uri :
    precedence: arg uri -> env MLFLOW_TRACKING_URI -> fallback local file:./mlruns
    Retourne l'uri effective.
    """
    if not uri:
        uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if not uri:
        uri = f"file:{os.path.abspath('./mlruns')}"
    try:
        mlflow.set_tracking_uri(uri)
        print(f"[INFO] mlflow tracking uri set to: {uri}")
    except Exception as e:
        print(f"[WARN] mlflow.set_tracking_uri failed ({e}); continuing with uri={uri}")
    return uri

def _mlflow_safe_set_experiment(name: str):
    try:
        mlflow.set_experiment(name)
        print(f"[INFO] mlflow experiment set to: {name}")
    except Exception as e:
        print(f"[WARN] mlflow.set_experiment failed: {e}")

def _mlflow_log_path(path: str):
    """Log an artifact path (file or dir) with safety checks."""
    try:
        if not os.path.exists(path):
            print(f"[WARN] artifact path does not exist, skip: {path}")
            return
        if os.path.isdir(path):
            mlflow.log_artifacts(path, artifact_path=os.path.basename(path))
        else:
            mlflow.log_artifact(path, artifact_path=os.path.dirname(path) or None)
    except Exception as e:
        print(f"[WARN] mlflow logging artifact failed for {path}: {e}")

def _ensure_parent_dir(p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# ---------------- CLI ----------------
@click.command()
@click.option("--encoded-folder", type=click.Path(exists=True, file_okay=False),
              default="data/encoded", show_default=True,
              help="Dossier contenant X_test.csv et y_test.csv.")
@click.option("--model", type=click.Choice(["lightgbm", "xgboost"]), default="lightgbm",
              show_default=True, help="Famille de modèle à analyser.")
@click.option("--model-folder", type=click.Path(exists=True, file_okay=False),
              default="models", show_default=True, help="Dossier où le modèle entraîné est sauvegardé.")
@click.option("--model-path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Chemin direct vers le .joblib (prioritaire sur --model/--model-folder).")
@click.option("--mlflow-uri", default=None, help="(optionnel) MLFLOW_TRACKING_URI ou laisser pour fallback local.")
@click.option("--experiment", default="regression_pipeline", show_default=True,
              help="Nom d'experience MLflow.")
@click.option("--run-id", default=None, help="(optionnel) rattacher à run existante (MLFLOW_RUN_ID alternative).")
@click.option("--save-metrics", default="metrics/analyse_metrics.json", show_default=True,
              help="Chemin pour sauvegarder métriques localement (JSON).")
def analyse_model(encoded_folder, model, model_folder, model_path, mlflow_uri, experiment, run_id, save_metrics):
    encoded_folder = Path(encoded_folder)
    model_folder = Path(model_folder)

    # mapping nom -> fichier par défaut
    default_files = {
        "lightgbm": "lgbm_model.joblib",
        "xgboost": "xgb_model.joblib",
    }
    if model_path is None:
        model_filename = default_files.get(model, f"{model}_model.joblib")
        model_path = model_folder / model_filename
    else:
        model_path = Path(model_path)

    X_test_path = encoded_folder / "X_test.csv"
    y_test_path = encoded_folder / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Je ne trouve pas X_test/y_test dans {encoded_folder}. Attendus: {X_test_path} et {y_test_path}"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Je ne trouve pas le modèle: {model_path}")

    print(f"[INFO] Encoded folder: {encoded_folder}")
    print(f"[INFO] Model path:     {model_path}")

    # MLflow setup (robuste)
    effective_uri = _mlflow_safe_set_tracking_uri(mlflow_uri)
    _mlflow_safe_set_experiment(experiment)

    # chargement
    model_obj = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path, sep=";", low_memory=False)
    y_test = pd.read_csv(y_test_path, sep=";", low_memory=False).values.ravel()

    # prédictions & metrics
    try:
        y_pred = model_obj.predict(X_test)
    except Exception as e:
        print(f"[FATAL] impossible de prédire avec le modèle: {e}")
        raise

    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics)

    # artefacts locaux (SHAP)
    shap_png = Path("exports/reg/shap_summary.png")
    _ensure_parent_dir(shap_png)
    try:
        # calcul SHAP (l'implémentation gère l'absence de shap)
        shap_summary_plot(model_obj, X_test, out_path=str(shap_png))
    except Exception as e:
        print(f"[WARN] SHAP summary plot impossible: {e}")

    residuals_png = None  # si tu veux activer plot_residuals, définis le chemin et la fonction correspondante

    # attach to run or create new run
    parent_run_id = run_id or os.getenv("MLFLOW_RUN_ID")
    try:
        if parent_run_id:
            run_ctx = mlflow.start_run(run_id=parent_run_id)
        else:
            run_ctx = mlflow.start_run(run_name=f"analyse_{model}")
    except Exception as e:
        print(f"[WARN] mlflow.start_run failed to use run_id='{parent_run_id}': {e}. Starting a fresh run.")
        run_ctx = mlflow.start_run(run_name=f"analyse_{model}")

    with run_ctx:
        # metrics (numeriques)
        to_log = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        try:
            if to_log:
                mlflow.log_metrics(to_log)
        except Exception as e:
            print(f"[WARN] mlflow.log_metrics failed: {e}")

        # tags
        try:
            mlflow.set_tag("stage", "evaluate")
            mlflow.set_tag("model_family", model)
            mlflow.set_tag("mlflow_tracking_uri", effective_uri)
        except Exception:
            pass

        # artefacts: shap, model, X_test, y_test
        try:
            if shap_png.exists():
                _mlflow_log_path(str(shap_png))
            if residuals_png and Path(residuals_png).exists():
                _mlflow_log_path(str(residuals_png))
            _mlflow_log_path(str(model_path))
            _mlflow_log_path(str(X_test_path))
            _mlflow_log_path(str(y_test_path))
        except Exception as e:
            print(f"[WARN] erreurs lors du log d'artefacts: {e}")

    # save metrics JSON locally (utile pour DVC metrics)
    try:
        Path(save_metrics).parent.mkdir(parents=True, exist_ok=True)
        with open(save_metrics, "w", encoding="utf-8") as fo:
            json.dump(metrics, fo, indent=2, ensure_ascii=False)
        print(f"[INFO] metrics JSON écrit: {save_metrics}")
    except Exception as e:
        print(f"[WARN] impossible d'ecrire metrics JSON: {e}")

    print("[INFO] Analyse terminée.")
    print("[INFO] MLflow run_id:", run_ctx.info.run_id if run_ctx else "n/a")

if __name__ == "__main__":
    analyse_model()
