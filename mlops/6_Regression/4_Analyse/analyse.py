# path: mlops/6_Regression/4_Analyse/analyse.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import click
import joblib
import pandas as pd
import mlflow

# ---- Utils (métriques/plots) ----
UTILS_DIR = Path(__file__).resolve().parent.parent / "3_UTILS"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from utils import compute_metrics, print_metrics, shap_summary_plot  # noqa: E402


def _ensure_parent_dir(p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option(
    "--encoded-folder",
    type=click.Path(exists=True, file_okay=False),
    default="data/encoded",
    show_default=True,
    help="Dossier contenant X_test.csv et y_test.csv.",
)
@click.option(
    "--model",
    type=click.Choice(["lightgbm", "xgboost"]),
    default="lightgbm",
    show_default=True,
    help="Famille de modèle à analyser.",
)
@click.option(
    "--model-folder",
    type=click.Path(exists=True, file_okay=False),
    default="models",
    show_default=True,
    help="Dossier où le modèle entraîné est sauvegardé.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Chemin direct vers le .joblib (prioritaire sur --model/--model-folder).",
)
def analyse_model(encoded_folder, model, model_folder, model_path):
    """
    Charge le modèle + X_test/y_test, calcule les métriques, génère des visuels,
    et loggue le tout dans MLflow (métriques + artefacts). Si MLFLOW_RUN_ID est
    définie, on se rattache à la même run; sinon, on en crée une nouvelle.
    """
    # --- Résolution des chemins ---
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
            f"Je ne trouve pas X_test/y_test dans {encoded_folder}. "
            f"Attendus: {X_test_path} et {y_test_path}"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Je ne trouve pas le modèle: {model_path}")

    # --- Chargement ---
    print(f"[INFO] Encoded folder: {encoded_folder}")
    print(f"[INFO] Model path:     {model_path}")
    model_obj = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path, sep=";")
    y_test = pd.read_csv(y_test_path, sep=";").values.ravel()

    # --- Prédictions & métriques ---
    y_pred = model_obj.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)  # doit renvoyer au moins rmse/MAE/R2…
    print_metrics(metrics)

    # --- Visuels / artefacts locaux ---
    shap_png = "exports/reg/shap_summary.png"
    _ensure_parent_dir(shap_png)
    try:
        shap_summary_plot(model_obj, X_test, out_path=shap_png)
    except Exception as e:
        print(f"[WARN] SHAP summary plot impossible: {e}")

    # Optionnel : si tu as une fonction qui sauvegarde les résidus
    # residuals_png = "exports/reg/residuals.png"
    # _ensure_parent_dir(residuals_png)
    # try:
    #     plot_residuals(y_test, y_pred, out_path=residuals_png)
    # except Exception as e:
    #     print(f"[WARN] Plot résidus impossible: {e}")
    residuals_png = None  # si tu actives ci-dessus, remplace None par le chemin

    # --- MLflow ---
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("regression_pipeline")

    parent_run_id = os.getenv("MLFLOW_RUN_ID")
    run_ctx = (
        mlflow.start_run(run_id=parent_run_id)
        if parent_run_id
        else mlflow.start_run(run_name=f"analyse_{model}")
    )
    with run_ctx:
        # métriques
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        # clé de comparaison normalisée
        rmse = metrics.get("rmse") or metrics.get("RMSE") or metrics.get("val_rmse")
        if rmse is not None:
            mlflow.log_metric("val_rmse", float(rmse))

        # tags
        mlflow.set_tag("stage", "evaluate")
        mlflow.set_tag("model_family", model)

        # artefacts
        if Path(shap_png).exists():
            mlflow.log_artifact(shap_png)
        if residuals_png and Path(residuals_png).exists():
            mlflow.log_artifact(residuals_png)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(X_test_path))
        mlflow.log_artifact(str(y_test_path))

    print("[INFO] Analyse terminée et logguée dans MLflow.")

