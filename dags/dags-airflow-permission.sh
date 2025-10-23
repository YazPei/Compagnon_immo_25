# 1) Donne lecture à tous les fichiers, et "traverse" aux dossiers
chmod -R a+rX dags
set -euo pipefail
: "${AIRFLOW_HOME:=/opt/airflow}"
airflow variables set SOURCE_MODE    "${SOURCE_MODE:-s3_public}"
airflow variables set S3_BUCKET      "${S3_BUCKET:-mon-bucket}"
airflow variables set S3_KEY         "${S3_KEY:-path/to/file.csv}"
airflow variables set S3_REGION      "${S3_REGION:-eu-west-1}"
airflow variables set S3_ENDPOINT    "${S3_ENDPOINT:-}"
airflow variables set AWS_PROFILE    "${AWS_PROFILE:-}"

# (au besoin, plus strict/verbeux)
find dags -type d -exec chmod 755 {} \;
find dags -type f -name "*.py" -exec chmod 644 {} \;

# 2) (optionnel) remet l’ownership à ton user
sudo chown -R "$USER":"$USER" dags

# 3) (optionnel mais sain) retire les CRLF dans les DAGs
find dags -type f -name "*.py" -exec sed -i 's/\r$//' {} \;

# 4) Redémarre Airflow pour rescanner
docker compose restart airflow

