# path: dags/compagnon_immo_stage.py
# WHY: DAG aligné S3 public/profil/DVC. Utilise ton import_data.py patché.
import os
import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator

PARIS = pendulum.timezone("Europe/Paris")
REPO = "/opt/airflow/repo"
PY = "python"

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
        cwd=cwd,
        env=env,
        execution_timeout=(pendulum.duration(minutes=timeout_min) if timeout_min else None),
    )

def build_import_cmd():
    source_mode = Variable.get("SOURCE_MODE", default_var="s3_public")  # s3_public|s3_profile|dvc
    s3_bucket  = Variable.get("S3_BUCKET", default_var=None)
    s3_key     = Variable.get("S3_KEY", default_var=None)
    s3_region  = Variable.get("S3_REGION", default_var="eu-west-1")
    s3_endpoint= Variable.get("S3_ENDPOINT", default_var=None)
    aws_profile= Variable.get("AWS_PROFILE", default_var=None)

    out_folder = f"{REPO}/data/incremental/{{{{ ds }}}}"
    cumul_csv  = f"{REPO}/data/df_sample.csv"
    chk_path   = "/opt/airflow/data/state/immo_checkpoint.parquet"
    date_col   = "date_vente"
    key_cols   = "id_transaction"

    if source_mode == "dvc":
        dvc_repo_url = Variable.get("DVC_REPO_URL", default_var=None)
        dvc_file_path= Variable.get("DVC_FILE_PATH", default_var=None)
        dvc_rev      = Variable.get("DVC_REV", default_var="main")
        assert dvc_repo_url and dvc_file_path, "DVC_REPO_URL/DVC_FILE_PATH requis en mode dvc"
        cmd = (
            f"{PY} {REPO}/mlops/1_import_donnees/import_data.py "
            f"--output-folder {out_folder} --cumulative-path {cumul_csv} "
            f"--checkpoint-path {chk_path} --date-column {date_col} --key-columns {key_cols} --sep ';' "
            f"--source-mode dvc --dvc-repo-url '{dvc_repo_url}' --dvc-path '{dvc_file_path}' --dvc-rev '{dvc_rev}'"
        )
        return cmd, {}

    assert s3_bucket and s3_key, "S3_BUCKET/S3_KEY requis en mode s3_*"
    base_cmd = (
        f"{PY} {REPO}/mlops/1_import_donnees/import_data.py "
        f"--output-folder {out_folder} --cumulative-path {cumul_csv} "
        f"--checkpoint-path {chk_path} --date-column {date_col} --key-columns {key_cols} --sep ';' "
        f"--source-mode s3 --s3-bucket '{s3_bucket}' --s3-key '{s3_key}' --s3-region '{s3_region}' "
    )
    if s3_endpoint:
        base_cmd += f"--s3-endpoint-url '{s3_endpoint}' "

    if source_mode == "s3_public":
        # Mode public: UNSIGNED + pas de creds requis
        return base_cmd + "--s3-anon", {"AWS_NO_SIGN_REQUEST": "1"}

    if source_mode == "s3_profile":
        env = {}
        if aws_profile:
            env["AWS_PROFILE"] = aws_profile
            env["AWS_SDK_LOAD_CONFIG"] = "1"
        return base_cmd, env

    raise ValueError(f"Unknown SOURCE_MODE={source_mode}")

with DAG(
    dag_id="compagnon_immo_stage",
    start_date=pendulum.datetime(2025, 9, 1, tz=PARIS),
    schedule="0 3 * * 1",
    catchup=False,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=10)},
    tags=["immo", "s3", "mlflow", "dvc"],
) as dag:
    init_dirs = bash_task("init_dirs", "mkdir -p /opt/airflow/data/state /opt/airflow/data/incremental", timeout_min=1)

    import_cmd, import_env = build_import_cmd()
    import_data = bash_task("import_donnees", import_cmd, env_extra=import_env, timeout_min=45)

    init_dirs >> import_data

