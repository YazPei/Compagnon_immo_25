# mlops/6_Regression/4_Analyse/analyse.py
import os
import sys
from pathlib import Path

import click
import joblib
import pandas as pd
import mlflow


UTILS_DIR = Path(__file__).resolve().parent.parent / "3_UTILS"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from utils import compute_metrics, print_metrics, plot_residuals, shap_summary_plot  # noqa: E402


def _ensure_dir(p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option('--encoded-folder', prompt='Dossier des fichiers encodés', type=click.Path(exists=True))
@click.option('--model', type=click.Choice(['lightgbm', 'xgboost']), prompt='Modèle à analyser')
def analyse_model(encoded_folder, model):
    """
    Lit le modèle + X_test/y_test, calcule les métriques, et LOGGUE dans MLflow :
    - toutes les métriques (dont rmse)
    - une copie sous la clé standardisée 'val_rmse' (pour la comparaison automatique)
    - artefacts (plot SHAP, résidus, fichiers)
    Si la variable d'env MLFLOW_RUN_ID est définie, on loggue dans la même run (train) ; sinon on crée une run.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("regression_pipeline")

    # chemins
    encoded_folder = Path(encoded_folder)
    model_path = encoded_folder / f'{model}_model.joblib'
    X_test_path = encoded_folder / 'X_test.csv'
    y_test_path = encoded_folder / 'y_test.csv'

    # charge
    model_obj = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path, sep=';')
    y_test = pd.read_csv(y_test_path, sep=';').values.ravel()

    # métriques
    y_pred = model_obj.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)  # doit contenir 'rmse' (ou adapte ci-dessous)
    print_metrics(metrics)

    # plots/artefacts
    _ensure_dir("exports/reg/shap_summary.png")
    shap_summary_plot(model_obj, X_test, out_path="exports/reg/shap_summary.png")
    # Optionnel : résidus si tu as une fonction qui sauvegarde un fichier (sinon garde juste l’affichage)
    # plot_residuals(y_test, y_pred, out_path="exports/reg/residuals.png")  # si tu veux un png

    # ---- MLflow logging ----
    parent_run_id = os.getenv("MLFLOW_RUN_ID")  # passé par Airflow pour rester dans la même run
    active_ctx = mlflow.start_run(run_id=parent_run_id) if parent_run_id else mlflow.start_run(run_name=f"analyse_{model}")
    try:
        # log toutes les métriques
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

        # normalise la clé de comparaison : 'val_rmse'
        rmse = metrics.get("rmse") or metrics.get("RMSE") or metrics.get("val_rmse")
        if rmse is not None:
            mlflow.log_metric("val_rmse", float(rmse))

        # tags utiles pour recherche
        mlflow.set_tag("stage", "evaluate")
        mlflow.set_tag("model_family", model)

        # artefacts
        mlflow.log_artifact("exports/reg/shap_summary.png")
        # if Path("exports/reg/residuals.png").exists():
        #     mlflow.log_artifact("exports/reg/residuals.png")
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(X_test_path))
        mlflow.log_artifact(str(y_test_path))
    finally:
        # on ne ferme pas la run parent si on l'a rouverte (start_run(run_id=...)) → mlflow gère le contexte
        active_ctx.__exit__(None, None, None)


if __name__ == '__main__':
    analyse_model()
