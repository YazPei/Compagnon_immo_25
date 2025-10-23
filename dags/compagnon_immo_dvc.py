# path: compagnon_immo_25/dags/compagnon_immo_dvc.py
import os
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

PARIS = pendulum.timezone("Europe/Paris")
REPO = "/opt/airflow/repo"  # monté par docker-compose

def dvc_task(task_id: str, cmd: str, minutes: int | None = None) -> BashOperator:
    env = os.environ.copy()
    # on forward MLFLOW_TRACKING_URI (Dagshub) & éventuels creds
    for k in ("MLFLOW_TRACKING_URI","MLFLOW_TRACKING_USERNAME","MLFLOW_TRACKING_PASSWORD","MLFLOW_TRACKING_TOKEN"):
        if os.getenv(k):
            env[k] = os.getenv(k)
    # execute dans le repo + fail-fast
    return BashOperator(
        task_id=task_id,
        bash_command=f"set -euo pipefail; cd {REPO} && {cmd}",
        env=env,
        execution_timeout=(pendulum.duration(minutes=minutes) if minutes else None),
    )

with DAG(
    dag_id="compagnon_immo_dvc",
    start_date=pendulum.datetime(2025, 9, 1, tz=PARIS),
    schedule="0 3 * * 0",   # chaque dimanche 03:00
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1, 
    tags=["immo","dvc","mlflow"],
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=10)},
    params={
        "force_retrain": Param(False, type="boolean", description="Forcer l'entraînement"),
        "run_note": Param("", type="string", description="Note libre du run"),
    },
) as dag:

    # Sanity + warm cache
    check_repo  = dvc_task("check_repo",  "git rev-parse --show-toplevel && dvc root && echo repo_ok", 2)
    dvc_pull    = dvc_task("dvc_pull",    "dvc pull -v || true", 15)

    # ===== Stages DVC (noms = dvc.yaml) =====
    dvc_import_data   = dvc_task("import_data",   "dvc repro import_data   -v", 30)
    dvc_preprocess    = dvc_task("preprocessing", "dvc repro preprocessing -v", 20)
    dvc_cluster       = dvc_task("clustering",    "dvc repro clustering    -v", 30)

    # branche régression
    dvc_encode        = dvc_task("encode",        "dvc repro encode        -v", 20)
    dvc_train         = dvc_task("train_lgbm",    "dvc repro train_lgbm    -v", 25)
    dvc_analyse       = dvc_task("analyse",       "dvc repro analyse       -v", 10)

    # branche séries temporelles
    dvc_splitst       = dvc_task("splitst",       "dvc repro splitst       -v", 20)
    dvc_decompose     = dvc_task("decompose",     "dvc repro decompose     -v", 15)
    dvc_train_sarimax = dvc_task("train_sarimax", "dvc repro train_sarimax -v", 40)
    dvc_evaluate      = dvc_task("evaluate",      "dvc repro evaluate      -v", 15)

    # (optionnel) push des artefacts en fin de run
    dvc_push          = dvc_task("dvc_push",      "dvc push -v || true", 20)

    # ===== Orchestration demandée =====
    check_repo >> dvc_pull

    # tronc commun
    chain_common = dvc_pull >> dvc_import_data >> dvc_preprocess >> dvc_cluster

    # branche 1 : regression
    chain_reg   = chain_common >> dvc_encode >> dvc_train >> dvc_analyse

    # branche 2 : séries
    chain_st    = chain_common >> dvc_splitst >> dvc_decompose >> dvc_train_sarimax >> dvc_evaluate

    # final: push (quand les deux branches ont fini)
    [chain_reg, chain_st] >> dvc_push

