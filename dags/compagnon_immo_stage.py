# path: dags/compagnon_immo_stage.py
import os
import pendulum
import requests
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PythonModel, log_model


# 1) Tracking côté DagsHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# 2) Auth via variables d’environnement (déjà exportées dans le container)
# MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD sont lues automatiquement par mlflow.

# 3) (Optionnel) Nom d’expérience par défaut
DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "compagnon_immo_prod")
mlflow.set_experiment(DEFAULT_EXPERIMENT)

# ------------------- CONFIG -------------------
PARIS = pendulum.timezone("Europe/Paris")
REPO = "/opt/airflow/repo"       # repo Git+DVC monté dans les conteneurs Airflow
PY = f"{REPO}/.venv/bin/python"  # interpreteur du venv; sinon "python"
DVC = "dvc"

MODEL_NAME = os.getenv("MODEL_NAME", "ImmoModel")
PREDICT_API_RELOAD_URL = os.getenv("PREDICT_API_RELOAD_URL", "http://predict-api:8000/reload")
PREDICT_API_RELOAD_URL = os.getenv("PREDICT_API_RELOAD_URL", "http://api:8000/api/v1/reload")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")  # forwarded dans l'env des tasks

TARGET_METRIC = os.getenv("TARGET_METRIC", "val_rmse")  # metric pour comparer candidat vs prod
BETTER_MODE = os.getenv("BETTER_MODE", "min").lower()   # "min" ou "max"

# ------------------- HELPERS -------------------
def bash_task(task_id: str, cmd: str, minutes: int | None = None):
    """Run a shell command inside the repo with a clean env."""
    env = os.environ.copy()
    if MLFLOW_URI:
        env["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    # set -euo pipefail to fail fast; cd into repo to keep relative paths valid
    return BashOperator(
        task_id=task_id,
        bash_command=f"set -euo pipefail; cd {REPO} && {cmd}",
        env=env,
        execution_timeout=(pendulum.duration(minutes=minutes) if minutes else None),
    )

def _ensure_registered_model(client: MlflowClient, name: str):
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name=name)

def _is_better(new: float, ref: float) -> bool:
    return (new > ref) if BETTER_MODE == "max" else (new < ref)

# ------------------- PYTHON TASKS -------------------
def train_v1(**ctx):
    """
    Entraine et loggue un candidat minimal (pyfunc) dans MLflow.
    Remplace par ton vrai train si besoin; ici on se concentre sur la mécanique MLflow.
    """
    class ConstantModel(PythonModel):
        def load_context(self, context): ...
        def predict(self, context, model_input):
            import numpy as np
            n = len(model_input) if hasattr(model_input, "__len__") else 1
            return np.zeros(n, dtype=float)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Immo/Train")
    with mlflow.start_run(run_name="train_v1") as run:
        run_id = run.info.run_id
        # log d'un modèle + une métrique cible (remplace par ta vraie metric)
        log_model(python_model=ConstantModel(), artifact_path="model")
        mlflow.log_metric(TARGET_METRIC, 200000)
        # stocker le run_id pour les tasks suivantes
        ctx["ti"].xcom_push(key="run_id", value=run_id)
        print(f"[train_v1] run_id={run_id}")

def register_v1(**ctx):
    """Crée une Model Version dans le registry à partir du run_id candidat."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    run_id = ctx["ti"].xcom_pull(key="run_id", task_ids="train_v1")
    if not run_id:
        raise RuntimeError("run_id manquant depuis train_v1")

    _ensure_registered_model(client, MODEL_NAME)
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        description="Candidate from Airflow pipeline",
    )
    ctx["ti"].xcom_push(key="model_version", value=mv.version)
    print(f"[register_v1] created version v{mv.version} for run {run_id}")

def compare_and_promote(**ctx):
    """
    Compare la metric du candidat vs l'alias 'production'.
    Si meilleur (selon BETTER_MODE), bascule l'alias 'production' sur la version candidate.
    Historise via stages: candidate -> Production, ancien prod -> Archived.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    run_id = ctx["ti"].xcom_pull(key="run_id", task_ids="train_v1")
    version = ctx["ti"].xcom_pull(key="model_version", task_ids="register_v1")
    if not run_id or not version:
        raise RuntimeError("run_id ou model_version manquant")

    # récupérer metric candidate
    candidate = mlflow.get_run(run_id)
    cand_val = candidate.data.metrics.get(TARGET_METRIC)
    if cand_val is None:
        raise RuntimeError(f"Metrique {TARGET_METRIC} absente sur la run {run_id}")
    cand_val = float(cand_val)

    # récupérer ref prod (si alias existe)
    try:
        prod_mv = client.get_model_version_by_alias(MODEL_NAME, "production")
        prod_run = mlflow.get_run(prod_mv.run_id)
        prod_val_raw = prod_run.data.metrics.get(TARGET_METRIC)
        if prod_val_raw is None:
            prod_val = float("inf") if BETTER_MODE == "min" else float("-inf")
        else:
            prod_val = float(prod_val_raw)
        prod_ver = prod_mv.version
    except Exception:
        prod_mv = None
        prod_val = float("inf") if BETTER_MODE == "min" else float("-inf")
        prod_ver = None

    print(f"[compare] candidate {TARGET_METRIC}={cand_val} vs prod {TARGET_METRIC}={prod_val} (mode={BETTER_MODE})")

    if _is_better(cand_val, prod_val):
        # place alias production sur la nouvelle version pour histporique
        client.set_registered_model_alias(MODEL_NAME, "production", str(version))
        client.set_model_version_tag(MODEL_NAME, version, "decision", "promoted")
        # transitions de stage (pour l'historique)
        if prod_mv:
            client.transition_model_version_stage(MODEL_NAME, prod_ver, stage="Archived")
        client.transition_model_version_stage(MODEL_NAME, version, stage="Production")
        print(f"[promote] v{version} -> Production (old v{prod_ver} archived)")
    else:
        client.set_model_version_tag(MODEL_NAME, version, "decision", "rejected")
        client.transition_model_version_stage(MODEL_NAME, version, stage="Staging")
        print(f"[reject] v{version} reste en Staging")

def notify_api_reload(**ctx):
    """Ping l'API de prediction pour recharger le modèle (optionnel)."""
    url = PREDICT_API_RELOAD_URL
    try:
        r = requests.post(url, timeout=8)
        r.raise_for_status()
        print("[notify] API reloaded OK")
    except Exception as e:
        print("[notify] reload failed:", e)

# ------------------- DAG -------------------
with DAG(
    dag_id="compagnon_immo_all",
    start_date=pendulum.datetime(2025, 9, 1, tz=PARIS),
    schedule="0 3 * * 0",  # dimanche 03:00
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,                  # (optionnel) évite les chevauchements
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=10)},
    tags=["immo", "mlops","registry"],
    params={
        "force_retrain": Param(False, type="boolean", description="Forcer l'entraînement"),
        "run_note": Param("", type="string", description="Note libre du run"),
    },
) as dag:

    # Sanity
    check_repo = bash_task("check_repo", "git rev-parse --show-toplevel && dvc root && echo repo_ok", 2)
    dvc_pull   = bash_task("dvc_pull", "dvc pull -v || true", 15)

    # === DVC PIPELINE (adapte les noms aux stages de ton dvc.yaml) ===
    dvc_import_data  = bash_task("dvc_import_data",  "dvc repro import_data -v", 30)        
    dvc_preprocess  = bash_task("dvc_preprocess",  "dvc repro preprocessing -v", 30)    
    dvc_cluster  = bash_task("dvc_cluster",  "dvc repro clustering -v", 30)
    dvc_encode   = bash_task("dvc_encode",   "dvc repro encode -v",     20)
    dvc_train    = bash_task("dvc_train",    "dvc repro train_lgbm -v", 20)
    dvc_analyse  = bash_task("dvc_analyse",  "dvc repro analyse -v || true", 10) 
    dvc_splitst  = bash_task("dvc_splitst",  "dvc repro splitst -v",    20)
    dvc_decompose  = bash_task("dvc_decompose",  "dvc repro decompose -v",    20)
    dvc_train_sarimax  = bash_task("dvc_train_sarimax",  "dvc repro train_sarimax -v",    20)    
    dvc_evaluate = bash_task("dvc_evaluate", "dvc repro evaluate -v || true",     15)  # idem

    # === MLflow registry flow (train/register/compare/promote/reload) ===
    t_train   = PythonOperator(task_id="train_v1", python_callable=train_v1)
    t_register= PythonOperator(task_id="register_v1", python_callable=register_v1)
    t_compare = PythonOperator(task_id="compare_and_promote", python_callable=compare_and_promote)
    t_reload  = PythonOperator(task_id="notify_api_reload", python_callable=notify_api_reload)

    dvc_push = bash_task("dvc_push", "dvc push -v || true", 20)

    # Orchestration
    check_repo >> dvc_pull
    dvc_pull >> dvc_import_data >> dvc_preprocess >> dvc_cluster >> dvc_encode >> dvc_train >> dvc_analyse 
    dvc_pull >> dvc_import_data  >> dvc_preprocess >> dvc_cluster >> dvc_splitst >> dvc_decompose >> dvc_train_sarimax >> dvc_evaluate    
    # ensuite la partie MLflow registry
    dvc_analyse >> t_train >> t_register >> t_compare >> t_reload >> dvc_push

