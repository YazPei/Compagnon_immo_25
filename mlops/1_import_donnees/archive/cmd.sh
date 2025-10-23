#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-public}" # public|profile
: "${BUCKET:?BUCKET required}"
: "${KEY:?KEY required}"
: "${REGION:?REGION required}"
DATE_COL="${DATE_COL:-}"
KEY_COLS="${KEY_COLS:-}"

mkdir -p data/state data/incremental


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
# remplace par ton endpoint réel
export AWS_S3_ENDPOINT="https://dagshub.com/api/v1/repo-buckets/s3/YazPei"
python3 - <<'PY'
import os, boto3
from botocore import UNSIGNED
from botocore.config import Config
s3=boto3.client("s3", endpoint_url=os.environ["AWS_S3_ENDPOINT"],
                config=Config(signature_version=UNSIGNED), region_name="us-east-1")
b="Compagnon_immo"; k="merged_sales_data.csv"
print("Listing sample:"); print(s3.list_objects_v2(Bucket=b, MaxKeys=10))
print("Head:", s3.head_object(Bucket=b, Key=k))
PY
