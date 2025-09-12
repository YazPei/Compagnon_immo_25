import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator

# --- Où est monté ton repo dans le conteneur Airflow ---
REPO = "/opt/airflow/repo" # change si tu as choisi /repo dans le compose
BASE = f"{REPO}/mlops" # tes dossiers 1_import_donnees, 2_dvc, ...

PARIS = pendulum.timezone("Europe/Paris")
PY = "python"

MLFLOW_URI = Variable.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ST_SUFFIX = Variable.get("ST_SUFFIX", "")
default_args = {"retries": 2, "retry_delay": pendulum.duration(minutes=10)}

def bash_task(task_id, cmd, timeout_min=None, env_extra=None, cwd=REPO):
env = {"MLFLOW_TRACKING_URI": MLFLOW_URI, "RUN_MODE": "full", "ST_SUFFIX": ST_SUFFIX}
if env_extra: env.update(env_extra)
return BashOperator(
task_id=task_id,
cwd=cwd,
bash_command=cmd,
env=env,
execution_timeout=pendulum.duration(minutes=timeout_min) if timeout_min else None,
)

with DAG(
dag_id="immo_stage_by_stage",
start_date=pendulum.datetime(2025, 9, 1, tz=PARIS),
schedule="0 3 * * *",
catchup=False,
default_args=default_args,
tags=["immo","stages","mlflow"],
) as dag:

# 0) Sanity (facultatif)
ping_mlflow = bash_task(
"ping_mlflow",
cmd='curl -sf ${MLFLOW_TRACKING_URI} > /dev/null && echo "MLflow OK" || (echo "MLflow KO"; exit 1)',
timeout_min=1,
)

# 1) Import des données
# dossier: mlops/1_import_donnees/
import_data = bash_task(
"import_donnees",
cmd=f'{PY} {BASE}/1_import_donnees/import_data.py',
timeout_min=20,
)

# 2) Étapes DVC (si tu as des utilitaires ici)
# dossier: mlops/2_dvc/
dvc_ops = bash_task(
"dvc_ops",
cmd=f'{PY} {BASE}/2_dvc/main.py', # ou {BASE}/2_dvc/run_dvc.sh
timeout_min=10,
)

# 3) Fusion géo + DVF
# dossier: mlops/3_fusion/
fusion_geo = bash_task(
"fusion_geo",
cmd=f'{PY} {BASE}/3_fusion/fusion_geo_dvf.py',
timeout_min=20,
)

# 4) Préprocessing (ton étape 4)
# dossier: mlops/preprocessing_4/
preprocessing = bash_task(
"preprocessing_4",
cmd=(
f'{PY} {BASE}/preprocessing_4/preprocessing.py '
f'--input-path data '
f'--output-folder1 data '
f'--output-folder2 data '
f'--run-date {{ {{ ds }} }}'
),
timeout_min=30,
)

# 5) Clustering
# dossier: mlops/5_clustering/
clustering = bash_task(
"clustering",
cmd=(
f'{PY} {BASE}/5_clustering/Clustering.py '
f'--input-path data/train_clean_ST.csv '
f'--output-path1 data/df_cluster.csv '
f'--output-path2 data/df_sales_clean_ST.csv'
),
timeout_min=20,
)

# 6) Régression (encoding → train → analyse)
# dossier: mlops/6_Regression/...
encode = bash_task(
"encode",
cmd=f'{PY} {BASE}/6_Regression/1_Encoding/encoding.py',
timeout_min=20,
)
train_lgbm = bash_task(
"train_lgbm",
cmd=f'{PY} {BASE}/6_Regression/2_LGBM/train_lgbm.py',
timeout_min=45,
)
analyse = bash_task( # optionnel
"analyse",
cmd=f'{PY} {BASE}/6_Regression/4_Analyse/analyse.py',
timeout_min=15,
)

# 7) Séries temporelles (split → decomp → sarimax → evaluate)
# dossier: mlops/7_Serie_temporelle/...
split = bash_task(
"split",
cmd=f'{PY} {BASE}/7_Serie_temporelle/1_SPLIT/load_split.py',
timeout_min=10,
)
decompose = bash_task(
"decompose",
cmd=(
f'{PY} {BASE}/7_Serie_temporelle/2_Decompose/seasonal_decomp.py '
f'--input-folder exports/st '
f'--output-folder exports/st '
f'--run-date {{ {{ ds }} }}'
),
timeout_min=20,
)
train_sarimax = bash_task(
"train_sarimax",
cmd=(
f'{PY} {BASE}/7_Serie_temporelle/3_SARIMAX/sarimax_train.py '
f'--input-folder exports/st '
f'--output-folder exports/st '
f'--run-date {{ {{ ds }} }}'
),
timeout_min=45,
)
evaluate = bash_task(
"evaluate",
cmd=f'{PY} {BASE}/7_Serie_temporelle/4_EVALUATE/evaluate_ST.py',
timeout_min=10,
)

# Dépendances (ordre exact)
ping_mlflow >> import_data >> dvc_ops >> fusion_geo >> preprocessing >> clustering \
>> encode >> train_lgbm >> analyse >> split >> decompose >> train_sarimax >> evaluate