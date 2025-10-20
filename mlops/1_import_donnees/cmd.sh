#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-public}" # public|profile
: "${BUCKET:?BUCKET required}"
: "${KEY:?KEY required}"
: "${REGION:?REGION required}"
DATE_COL="${DATE_COL:-}"
KEY_COLS="${KEY_COLS:-}"

mkdir -p data/state data/incremental

if [[ "$MODE" == "public" ]]; then
  exec python3 import_data.py \
    --source-mode s3 \
    --s3-anon \
    --s3-bucket "$BUCKET" \
    --s3-key "$KEY" \
    --s3-region "$REGION" \
    --output-folder data/incremental \
    --cumulative-path data/df_sample.csv \
    --checkpoint-path data/state/checkpoint.parquet \
    --date-column "$DATE_COL" \
    --key-columns "$KEY_COLS" \
    --sep ";"
else
  : "${AWS_PROFILE:?AWS_PROFILE required for MODE=profile}"
  exec python3 import_data.py \
    --source-mode s3 \
    --s3-bucket "$BUCKET" \
    --s3-key "$KEY" \
    --s3-region "$REGION" \
    --output-folder data/incremental \
    --cumulative-path data/df_sample.csv \
    --checkpoint-path data/state/checkpoint.parquet \
    --date-column "$DATE_COL" \
    --key-columns "$KEY_COLS" \
    --sep ";"
fi
