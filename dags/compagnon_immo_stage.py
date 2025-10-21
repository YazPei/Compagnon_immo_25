import os
import pendulum
import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PythonModel, log_model

PARIS = pendulum.timezone("Europe/Paris")
REPO = "/opt/airflow/repo"
PY = "python"

# --------- Utils Bash (import 10%) ----------
def bash_task(task_id, cmd, timeout_min=None, env_extra=None, cwd=REPO):
    env = os.environ.copy()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        env["MLFLOW_TRACKING_URI"] = mlflow_uri
    if env_extra:
        env.update(env_extra)
    return BashOperator(
        task_id=task_id,
        bash_command=cmd,
        env=env,
        cwd=cwd,  # si non supporté par ta version, fais: bash_command=f"cd {cwd} && {cmd}"
        execution_timeout=(pendulum.duration(minutes=timeout_min) if timeout_min else None),
    )

def build_import_cmd():
    # ----- Variables Airflow (Admin > Variables)
    source_mode = Variable.get("SOURCE_MODE", default_var="s3_public")  # s3_public|s3_profile|dvc
    s3_bucket   = Variable.get("S3_BUCKET",  default_var=None)
    s3_key      = Variable.get("S3_KEY",     default_var=None)
    s3_region   = Variable.get("S3_REGION",  default_var="us-east-1")
    s3_endpoint = Variable.get("S3_ENDPOINT",default_var=None)
    aws_profile = Variable.get("AWS_PROFILE",default_var=None)

    NUM_SLICES = int(Variable.get("NUM_SLICES", default_var="10"))  # 10% ⇒ 10

    out_folder = f"{REPO}/data/incremental/{{{{ ds }}}}"
    cumul_csv  = f"{REPO}/data/df_sample.csv"
    chk_path   = "/opt/airflow/data/state/immo_checkpoint.parquet"
    date_col   = "date"
    key_cols   = "idannonce"

    # semaine ISO % NUM_SLICES  → tranche déterministe 0..N-1
    slice_expr = "{{ (data_interval_start.strftime('%V') | int) % " + str(NUM_SLICES) + " }}"

    if source_mode == "dvc":
        dvc_repo_url  = Variable.get("DVC_REPO_URL",  default_var=None)
        dvc_file_path = Variable.get("DVC_FILE_PATH", default_var=None)
        dvc_rev       = Variable.get("DVC_REV",       default_var="main")
        assert dvc_repo_url and dvc_file_path, "DVC_REPO_URL/DVC_FILE_PATH requis en mode dvc"
        cmd = (
            f"{PY} {REPO}/mlops/1_import_donnees/import_data.py "
            f"--output-folder {out_folder} "
            f"--cumulative-path {cumul_csv} "
            f"--checkpoint-path {chk_path} "
            f"--date-column {date_col} "
            f"--key-columns {key_cols} "
            f"--sep ';' "
            f"--source-mode dvc "
            f"--dvc-repo-url '{dvc_repo_url}' "
            f"--dvc-path '{dvc_file_path}' "
            f"--dvc-rev '{dvc_rev}' "
            f"--num-slices {NUM_SLICES} "
            f"--slice-index {slice_expr} "
            f"--append-only --dedup-duckdb"
        )
        return cmd, {}

    # Modes S3
    assert s3_bucket and s3_key, "S3_BUCKET/S3_KEY requis en mode s3_*"
    base_cmd = (
        f"{PY} {REPO}/mlops/1_import_donnees/import_data.py "
        f"--output-folder {out_folder} "
        f"--cumulative-path {cumul_csv} "
        f"--checkpoint-path {chk_path} "
        f"--date-column {date_col} "
        f"--key-columns {key_cols} "
        f"--sep ';' "
        f"--source-mode s3 "
        f"--s3-bucket '{s3_bucket}' "
        f"--s3-key '{s3_key}' "
        f"--s3-region '{s3_region}' "
        f"--num-slices {NUM_SLICES} "
        f"--slice-index {slice_expr} "
        f"--append-only --dedup-duckdb "
    )
    if s3_endpoint:
        base_cmd += f"--s3-endpoint-url '{s3_endpoint}' "

    if source_mode == "s3_public":
        return base_cmd + "--s3-anon", {"AWS_NO_SIGN_REQUEST": "1"}

    if source_mode == "s3_profile":
        env = {}
        if aws_profile:
            env["AWS_PROFILE"] = aws_profile
            env["AWS_SDK_LOAD_CONFIG"] = "1"
        return base_cmd, env

    raise ValueError(f"Unknown SOURCE_MODE={source_mode}")


# --------- Python callables (MLflow train/register/compare/reload) ----------
MODEL_NAME = Variable.get("MODEL_NAME", default_var="ImmoModel")
PREDICT_API_RELOAD_URL = Variable.get("PREDICT_API_RELOAD_URL", default_var="http://predict-api:8000/reload")
TARGET_METRIC = Variable.get("TARGET_METRIC", default_var="val_rmse")  # configurable
BETTER_MODE = Variable.get("BETTER_MODE", default_var="min")  # "min" or "max"

def _ensure_registered_model(client: MlflowClient, name: str):
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name=name)

def _is_better(new: float, ref: float) -> bool:
    if BETTER_MODE.lower() == "max":
        return new > ref
    return new < ref  # default "min"

def train_v1(**ctx):
    """
    Entraîne un modèle candidat et loggue dans MLflow.
    Pour être autonome (sans sklearn), on loggue un pyfunc minimal.
    """
    class ConstantModel(PythonModel):
        def load_context(self, context):
            pass
        def predict(self, context, model_input):
            # renvoie 0.0 → démo. Remplace par ton vrai modèle.
            import numpy as np
            import pandas as pd
            n = len(model_input) if hasattr(model_input, "__len__") else 1
            return np.zeros(n, dtype=float)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Immo/Train")
    with mlflow.start_run(run_name="train_v1") as run:
        run_id = run.info.run_id

        # calcule métrique (rmse, mape, etc.)
        ctx['ti'].xcom_push(key="run_id", value=run_id)
   

def register_v1(**ctx):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    run_id = ctx['ti'].xcom_pull(key="run_id", task_ids="train_v1")
    _ensure_registered_model(client, MODEL_NAME)
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        description="Candidate from Airflow pipeline"
    )
    ctx['ti'].xcom_push(key="model_version", value=mv.version)

def compare_and_promote(**ctx):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    run_id = ctx['ti'].xcom_pull(key="run_id", task_ids="train_v1")
    version = ctx['ti'].xcom_pull(key="model_version", task_ids="register_v1")

    candidate = mlflow.get_run(run_id)
    cand_val = candidate.data.metrics.get(TARGET_METRIC)
    if cand_val is None:
        raise RuntimeError(f"Métrique {TARGET_METRIC} absente sur la run candidate {run_id}")
    cand_val = float(cand_val)

    # PROD via alias "production" (si absent, le candidat gagne par défaut)
    try:
        prod_mv = client.get_model_version_by_alias(MODEL_NAME, "production")
        prod_run = mlflow.get_run(prod_mv.run_id)
        prod_val = float(prod_run.data.metrics.get(TARGET_METRIC, "inf" if BETTER_MODE=="min" else "-inf"))
    except Exception:
        prod_mv = None
        prod_val = float("inf") if BETTER_MODE=="min" else float("-inf")

    print(f"[compare] candidate {TARGET_METRIC}={cand_val} vs prod {TARGET_METRIC}={prod_val} (mode={BETTER_MODE})")

    if _is_better(cand_val, prod_val):
        # place l'alias production sur la nouvelle version
        client.set_registered_model_alias(MODEL_NAME, "production", str(version))
        client.set_model_version_tag(MODEL_NAME, version, "decision", "promoted")
        # optionnel: stage management pour historique
        if prod_mv:
            client.transition_model_version_stage(MODEL_NAME, prod_mv.version, stage="Archived")
        client.transition_model_version_stage(MODEL_NAME, version, stage="Production")
        print(f"[promote] v{version} -> production")
    else:
        client.set_model_version_tag(MODEL_NAME, version, "decision", "rejected")
        client.transition_model_version_stage(MODEL_NAME, version, stage="Staging")
        print(f"[reject] v{version} reste en Staging")
        
def call_analyse(**ctx):
    run_id = ctx['ti'].xcom_pull(key="run_id", task_ids="train_v1")
    if not run_id:
        raise RuntimeError("run_id manquant depuis train_v1")
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_RUN_ID"] = run_id

    # adapte le dossier encodé + modèle
    encoded_folder = f"{REPO}/mlops/6_Regression/1_Encoding/exports"  # ← mets le bon chemin
    model = "lightgbm"                                                # ← ou "xgboost"

    import subprocess, sys
    subprocess.check_call([
        sys.executable,
        f"{REPO}/mlops/6_Regression/4_Analyse/analyse.py",
        "--encoded-folder", encoded_folder,
        "--model", model,
    ])
def notify_api_reload(**ctx):
    url = PREDICT_API_RELOAD_URL
    try:
        r = requests.post(url, timeout=8)
        r.raise_for_status()
        print("[ok] API reloaded")
    except Exception as e:
        print("[warn] reload failed:", e)
        
def call_evaluate(**ctx):
    run_id = ctx['ti'].xcom_pull(key="run_id", task_ids="train_v1")
    if not run_id:
        raise RuntimeError("run_id manquant depuis train_v1")
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_RUN_ID"] = run_id
    import subprocess, sys
    subprocess.check_call([sys.executable, f"{REPO}/7_Serie_temporelle/4_EVALUATE/evaluate_ST.py"])        

# ---------------- DAG unique ----------------
with DAG(
    dag_id="compagnon_immo_pipeline",
    start_date=pendulum.datetime(2025, 9, 1, tz=PARIS),
    schedule="0 3 * * 0",  # Dimanche 03:00 Paris
    catchup=False,
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=10)},
    tags=["immo", "ingest", "mlflow", "registry", "deploy"],
) as dag:

    # Dossiers persistants
    init_dirs = bash_task(
        "init_dirs",
        cmd="mkdir -p /opt/airflow/data/state /opt/airflow/data/incremental && echo 'dirs ok'",
        timeout_min=1,
    )

    # Import 10%
    import_cmd, import_env = build_import_cmd()
    t_import = bash_task(
        "import_10pct",
        cmd=import_cmd,
        env_extra=import_env,
        timeout_min=60,
    )
    
    
        # 2) Étapes DVC (si utiles)
    dvc_ops = bash_task(
        "dvc_ops",
        cmd=f"{PY} {BASE}/2_dvc/main.py",
        timeout_min=10,
    )
        # 4) Préprocessing (ton étape 4)
    preprocessing = bash_task(
        "preprocessing_4",
        cmd=(
            f"{PY} {BASE}/preprocessing_4/preprocessing.py "
            f"--input-path data "
            f"--output-path data "
            f"--run-date {{{{ ds }}}}"
        ),
        timeout_min=30,
    )
    
    clustering = bash_task("clustering", cmd=(
            f"{PY} {BASE}/5_clustering/Clustering.py "
            f"--input-path data/train_clean_ST.csv "
            f"--output-path1 data/df_cluster.csv "
            f"--output-path2 data/df_sales_clean_ST.csv"
        ),
        timeout_min=20,
    )
    # 6) Régression (encoding → train → analyse)
    encode = bash_task(
        "encode",
        cmd=f"{PY} {BASE}/6_Regression/1_Encoding/encoding.py",
        timeout_min=20,
    )
    # Train / Register / Compare / Reload
    
    t_train   = PythonOperator(task_id="train_v1", python_callable=train_v1)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=call_evaluate)
    t_register= PythonOperator(task_id="register_v1", python_callable=register_v1)
    t_compare = PythonOperator(task_id="compare_and_promote", python_callable=compare_and_promote)
    t_reload  = PythonOperator(task_id="notify_api_reload", python_callable=notify_api_reload)

    # Orchestration
    init_dirs >> t_import >> t_train >> t_analyse >> t_register >> t_compare >> t_reload
