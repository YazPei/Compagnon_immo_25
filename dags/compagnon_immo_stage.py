# --- path: airflow/dags/stage.py
import os
import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator

REPO = "/opt/airflow/repo"
BASE = f"{REPO}/mlops"

PARIS = pendulum.timezone("Europe/Paris")
PY = "python"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
ST_SUFFIX = Variable.get("ST_SUFFIX", "")
default_args = {"retries": 2, "retry_delay": pendulum.duration(minutes=10)}

# --- DVC/Dagshub variables (Airflow Variables) ---
DVC_REPO_URL = Variable.get("DVC_REPO_URL", default_var=None)      # ex: https://dagshub.com/<user>/<repo>
DVC_FILE_PATH = Variable.get("DVC_FILE_PATH", default_var=None)    # ex: data/raw/dvc_data.csv
DVC_REV = Variable.get("DVC_REV", default_var="main")
DAGSHUB_USERNAME = Variable.get("DAGSHUB_USERNAME", default_var=None)
DAGSHUB_TOKEN = Variable.get("DAGSHUB_TOKEN", default_var=None)

def bash_task(task_id, cmd, timeout_min=None, env_extra=None, cwd=REPO):
    env = os.environ.copy()
    if MLFLOW_URI:
        env["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    env["RUN_MODE"] = "full"
    env["ST_SUFFIX"] = ST_SUFFIX
    # DVC/Dagshub creds pour dvc.api.open()
    if DVC_REPO_URL:
        env["DVC_REPO_URL"] = DVC_REPO_URL
    if DVC_FILE_PATH:
        env["DVC_FILE_PATH"] = DVC_FILE_PATH
    if DVC_REV:
        env["DVC_REV"] = DVC_REV
    if DAGSHUB_USERNAME:
        env["DAGSHUB_USERNAME"] = DAGSHUB_USERNAME
    if DAGSHUB_TOKEN:
        env["DAGSHUB_TOKEN"] = DAGSHUB_TOKEN
    # utile si tu fais des imports relatifs
    env["PYTHONPATH"] = REPO

    if env_extra:
        env.update(env_extra)
    return BashOperator(
        task_id=task_id,
        cwd=cwd,
        bash_command=cmd,
        env=env,
        execution_timeout=(pendulum.duration(minutes=timeout_min) if timeout_min else None),
    )

with DAG(
    dag_id="immo_stage_by_stage",
    start_date=pendulum.datetime(2025, 9, 1, tz=PARIS),
    schedule="0 3 * * 1",
    catchup=False,
    default_args=default_args,
    tags=["immo", "stages", "mlflow"],
) as dag:

    # 0) Sanity (MLflow)
    ping_mlflow = bash_task(
        "ping_mlflow",
        cmd='curl -sf "${MLFLOW_TRACKING_URI}" > /dev/null && echo "MLflow OK" || (echo "MLflow KO"; exit 1)',
        timeout_min=1,
    )

    # 0bis) Prépare les dossiers persistants (checkpoint + incremental)
    init_dirs = bash_task(
        "init_dirs",
        cmd="mkdir -p /opt/airflow/data/state /opt/airflow/data/incremental && echo 'dirs ok'",
        timeout_min=1,
    )

    # 1) Import des données — **MODE DVC/Dagshub**
    # NOTE: pour conserver la templating Jinja d'Airflow dans une f-string Python,
    # on met des quadruples accolades: {{{{ ds }}}}
    import_data = bash_task(
        "import_donnees",
        cmd=(
            f"{PY} {BASE}/1_import_donnees/import_data.py "
            f"--output-folder {REPO}/data/incremental/{{{{ ds }}}} "
            f"--cumulative-path {REPO}/data/df_sample.csv "
            f"--checkpoint-path /opt/airflow/data/state/immo_checkpoint.parquet "
            f"--date-column date_vente "
            f"--key-columns id_transaction "
            f"--sep ';' "
            f"--chunk-size 200000 "
            f"--dvc-repo-url \"$DVC_REPO_URL\" "
            f"--dvc-path \"$DVC_FILE_PATH\" "
            f"--dvc-rev \"$DVC_REV\" "
        ),
        timeout_min=30,
    )

    # 2) Étapes DVC (si utiles)
    dvc_ops = bash_task(
        "dvc_ops",
        cmd=f"{PY} {BASE}/2_dvc/main.py",
        timeout_min=10,
    )

    # 3) Fusion géo + DVF
    fusion_geo = bash_task(
        "fusion_geo",
        cmd=f"{PY} {BASE}/3_fusion/fusion_geo_dvf.py",
        timeout_min=20,
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

    # 5) Clustering
    clustering = bash_task(
        "clustering",
        cmd=(
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
    train_lgbm = bash_task(
        "train_lgbm",
        cmd=f"{PY} {BASE}/6_Regression/2_LGBM/train_lgbm.py",
        timeout_min=45,
    )
    analyse = bash_task(
        "analyse",
        cmd=f"{PY} {BASE}/6_Regression/4_Analyse/analyse.py",
        timeout_min=15,
    )

    # 7) Séries temporelles
    split = bash_task(
        "split",
        cmd=f"{PY} {BASE}/7_Serie_temporelle/1_SPLIT/load_split.py",
        timeout_min=10,
    )
    decompose = bash_task(
        "decompose",
        cmd=(
            f"{PY} {BASE}/7_Serie_temporelle/2_Decompose/seasonal_decomp.py "
            f"--input-folder exports/st "
            f"--output-folder exports/st "
            f"--run-date {{{{ ds }}}}"
        ),
        timeout_min=20,
    )
    train_sarimax = bash_task(
        "train_sarimax",
        cmd=(
            f"{PY} {BASE}/7_Serie_temporelle/3_SARIMAX/sarimax_train.py "
            f"--input-folder exports/st "
            f"--output-folder exports/st "
            f"--run-date {{{{ ds }}}}"
        ),
        timeout_min=45,
    )
    evaluate = bash_task(
        "evaluate",
        cmd=f"{PY} {BASE}/7_Serie_temporelle/4_EVALUATE/evaluate_ST.py",
        timeout_min=10,
    )

    # Orchestration
    (
        ping_mlflow
        >> init_dirs
        >> import_data
        >> dvc_ops
        >> fusion_geo
        >> preprocessing
        >> clustering
        >> encode
        >> train_lgbm
        >> analyse
        >> split
        >> decompose
        >> train_sarimax
        >> evaluate
    )

