#!/usr/bin/env bash
set -euo pipefail

# charge ton env (adapter le chemin si besoin)
[ -f .env ] && . .env || true
[ -f "/home/vboxuser/Compagnon_new/.dagshub.env" ] && . "/home/vboxuser/Compagnon_new/.dagshub.env" || true

# sanity checks utiles
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID manquant}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY manquant}"

# lance avec le python du venv
./.venv/bin/python mlops/1_import_donnees/import_data.py \
  --source-mode s3 \
  --s3-endpoint-url "https://dagshub.com/api/v1/repo-buckets/s3/YazPei" \
  --s3-bucket "Compagnon_immo_25" \
  --s3-key "merged_sales_data.csv" \
  --s3-region "us-east-1" \
  --output-folder data/incremental \
  --cumulative-path data/df_sample.csv \
  --checkpoint-path data/state/checkpoint.parquet \
  --date-column "date" \
  --key-columns "idannonce" \
  --sep ";"

