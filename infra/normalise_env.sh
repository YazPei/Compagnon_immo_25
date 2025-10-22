#!/usr/bin/env bash
set -euo pipefail
ENV_FILE="${1:-.env}"
[ -f "$ENV_FILE" ] || { echo "ERR: $ENV_FILE introuvable"; exit 1; }

# 1) S3_ENDPOINT (priorité à AWS_S3_ENDPOINT puis AWS_ENDPOINT_URL_S3 si S3_ENDPOINT vide)
S3_EP="$(grep -E '^S3_ENDPOINT=' "$ENV_FILE" | cut -d= -f2- || true)"
if [ -z "${S3_EP:-}" ]; then
  AWS_S3_EP="$(grep -E '^AWS_S3_ENDPOINT=' "$ENV_FILE" | cut -d= -f2- || true)"
  AWS_EP_S3="$(grep -E '^AWS_ENDPOINT_URL_S3=' "$ENV_FILE" | cut -d= -f2- || true)"
  if [ -n "${AWS_S3_EP:-}" ]; then
    echo "S3_ENDPOINT=${AWS_S3_EP}" >> "$ENV_FILE"
  elif [ -n "${AWS_EP_S3:-}" ]; then
    echo "S3_ENDPOINT=${AWS_EP_S3}" >> "$ENV_FILE"
  fi
fi

# 2) BUCKET depuis DAGSHUB_BUCKET si BUCKET vide
BUCKET_V="$(grep -E '^BUCKET=' "$ENV_FILE" | cut -d= -f2- || true)"
if [ -z "${BUCKET_V:-}" ]; then
  DGH_B="$(grep -E '^DAGSHUB_BUCKET=' "$ENV_FILE" | cut -d= -f2- || true)"
  [ -n "${DGH_B:-}" ] && echo "BUCKET=${DGH_B}" >> "$ENV_FILE"
fi

# 3) REGION depuis AWS_DEFAULT_REGION si REGION vide
REGION_V="$(grep -E '^REGION=' "$ENV_FILE" | cut -d= -f2- || true)"
if [ -z "${REGION_V:-}" ]; then
  AWS_REG="$(grep -E '^AWS_DEFAULT_REGION=' "$ENV_FILE" | cut -d= -f2- || true)"
  [ -n "${AWS_REG:-}" ] && echo "REGION=${AWS_REG}" >> "$ENV_FILE"
fi

# 4) SOURCE_MODE défaut si absent (s3_profile pour DagsHub privé)
grep -qE '^SOURCE_MODE=' "$ENV_FILE" || echo "SOURCE_MODE=s3_profile" >> "$ENV_FILE"

echo "[ok] normalisation .env terminée"

