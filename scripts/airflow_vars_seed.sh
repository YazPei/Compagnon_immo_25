set -euo pipefail
: "${AIRFLOW_HOME:=/opt/airflow}"
airflow variables set SOURCE_MODE    "${SOURCE_MODE:-s3_public}"
airflow variables set S3_BUCKET      "${S3_BUCKET:-mon-bucket}"
airflow variables set S3_KEY         "${S3_KEY:-path/to/file.csv}"
airflow variables set S3_REGION      "${S3_REGION:-eu-west-1}"
airflow variables set S3_ENDPOINT    "${S3_ENDPOINT:-}"
airflow variables set AWS_PROFILE    "${AWS_PROFILE:-}"
