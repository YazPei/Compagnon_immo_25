import os
import pendulum
# import mlflow  # facultatif si tu ne l'utilises pas dans le DAG lui-même

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable

# Repo monté dans le conteneur Airflow (voir docker-compose)
REPO = "/opt/airflow/repo"
BASE = f"{REPO}/mlops"

PARIS = pendulum.timezone("Europe/Paris")
PY = "python"

# Optionnel : récup depuis .env passé au conteneur Airflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")
ST_SUFFIX = Variable.get("ST_SUFFIX", default_var="")

default_args = {"retries": 2, "retry_delay": pendulum.duration(minutes=10)}

def bash_task(task_id, cmd, timeout_min=None, env_extra=None, cwd=REPO):
    env = os.environ.copy()
    if MLFLOW_URI:
        env["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    env["RUN_MODE"] = "full"
    env["ST_SUFFIX"] = ST_SUFFIX
    if env_extra:
        env.update(env_extra)

    return BashOperator(
        task_id=task_id,
        cwd=cwd,                # OK avec Airflow 2.7+
        bash_command=cmd,       # ⚠️ pas de f-string si tu utilises des {{ macros }}
        env=env,
        execution_timeout=pendulum.duration(minutes=timeout_min) if timeout_min else None,
    )

with DAG(
    dag_id="immo_stage_by_stage",
    start_date=pendulum.datetime(2024, 9, 1, tz=PARIS),  # une date passée
    schedule="0 3 * * *",
    catchup=False,
    default_args=default_args,
    tags=["immo", "stages", "mlflow"],
) as dag:

    ping_mlflow = bash_task(
        "ping_mlflow",
        cmd='curl -sf ${MLFLOW_TRACKING_URI} > /dev/null && echo "MLflow OK" || (echo "MLflow KO"; exit 1)',
        timeout_min=1,
    )

    import_data = bash_task(
        "import_donnees",
        cmd=f"{PY} {BASE}/1_import_donnees/import_data.py",
        timeout_min=20,
    )

    dvc_ops = bash_task(
        "dvc_ops",
        cmd=f"{PY} {BASE}/2_dvc/main.py",  # adapte si c'est un .sh
        timeout_min=10,
    )

    fusion_geo = bash_task(
        "fusion_geo",
        cmd=f"{PY} {BASE}/3_fusion/fusion.py",
        timeout_min=20,
    )

    preprocessing = bash_task(
        "preprocessing_4",
        # ⚠️ ICI pas d'f-string pour laisser Airflow templater {{ ds }} si tu l'utilises
        cmd="python mlops/preprocessing_4/preprocessing.py --input-path data --output-path data",
        timeout_min=30,
    )

    clustering = bash_task(
        "clustering",
        cmd="python mlops/5_clustering/Clustering.py "
            "--input-path data/train_clean.csv "
            "--output-path1 data/df_cluster.csv "
            "--output-path2 data/df_sales_clean_ST.csv",
        timeout_min=20,
    )

    encode = bash_task(
        "encode",
        cmd="python mlops/6_Regression/1_Encoding/encoding.py",
        timeout_min=20,
    )

    train_lgbm = bash_task(
        "train_lgbm",
        cmd="python mlops/6_Regression/2_LGBM/train_lgbm.py",
        timeout_min=45,
    )

    analyse = bash_task(
        "analyse",
        cmd="python mlops/6_Regression/4_Analyse/analyse.py",
        timeout_min=15,
    )

    split = bash_task(
        "split",
        cmd="python mlops/7_Serie_temporelle/1_SPLIT/load_split.py",
        timeout_min=10,
    )

    decompose = bash_task(
        "decompose",
        cmd="python mlops/7_Serie_temporelle/2_Decompose/seasonal_decomp.py "
            "--input-folder exports/st --output-folder exports/st",
        timeout_min=20,
    )

    train_sarimax = bash_task(
        "train_sarimax",
        cmd="python mlops/7_Serie_temporelle/3_SARIMAX/sarimax_train.py "
            "--input-folder exports/st --output-folder exports/st",
        timeout_min=45,
    )

    evaluate = bash_task(
        "evaluate",
        cmd="python mlops/7_Serie_temporelle/4_EVALUATE/evaluate_ST.py",
        timeout_min=10,
    )

    (ping_mlflow >> import_data >> dvc_ops >> fusion_geo >> preprocessing >>
     clustering >> encode >> train_lgbm >> analyse >> split >> decompose >>
     train_sarimax >> evaluate)

