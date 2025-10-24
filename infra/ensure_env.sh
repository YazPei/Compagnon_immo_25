#!/usr/bin/env bash
set -euo pipefail
ENV_FILE="${1:-.env}"
touch "$ENV_FILE"

add_if_missing () {
  local key="$1"; shift
  local val="$*"
  if ! grep -qE "^${key}=" "$ENV_FILE"; then
    echo "${key}=${val}" >> "$ENV_FILE"
    echo "[env] added ${key}"
  else
    echo "[env] kept  ${key}"
  fi
}

add_if_missing AIRFLOW_UID "50000"

if ! grep -qE "^AIRFLOW_FERNET_KEY=" "$ENV_FILE"; then
  add_if_missing AIRFLOW_FERNET_KEY "$(openssl rand -base64 32 | tr -d '\n')"
fi
if ! grep -qE "^AIRFLOW_SECRET_KEY=" "$ENV_FILE"; then
  add_if_missing AIRFLOW_SECRET_KEY "$(openssl rand -base64 32 | tr -d '\n')"
fi


# MLflow / registry
add_if_missing MODEL_NAME "ImmoModel"
add_if_missing TARGET_METRIC "val_rmse"
add_if_missing BETTER_MODE "min"
add_if_missing PREDICT_API_RELOAD_URL "http://predict-api:8000/reload"
add_if_missing AWS_S3_ENDPOINT "https://dagshub.com/api/v1/repo-buckets/s3/YazPei"
add_if_missing BUCKET "Compagnon_immo_25"
add_if_missing KEY "merged_sales_data.csv"
add_if_missing REGION "us-east-1"
add_if_missing AWS_ACCESS_KEY_ID "$MLFLOW_TRACKING_PASSWORD"
add_if_missing AWS_SECRET_ACCESS_KEY "$MLFLOW_TRACKING_PASSWORD"	

# Ne touche PAS à MLFLOW_TRACKING_URI si déjà présent (ton Dagshub)
if ! grep -qE "^MLFLOW_TRACKING_URI=" "$ENV_FILE"; then
  add_if_missing MLFLOW_TRACKING_URI "http://mlflow:5050"
fi

echo "[env] done -> $ENV_FILE"

