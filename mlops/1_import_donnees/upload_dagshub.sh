#!/usr/bin/env bash
set -euo pipefail

: "${HOME:?}"
: "${1:?Usage: scripts/upload_dagshub.sh <local_file> [dest_key]}"

LOCAL_FILE="$1"
DEST_KEY="${2:-path/in/bucket/$(basename "$LOCAL_FILE")}"

# charge l'env
source "$HOME/.dagshub.env"

if [[ ! -f "$LOCAL_FILE" ]]; then
  echo "Fichier introuvable: $LOCAL_FILE" >&2; exit 2
fi

echo "Endpoint: $AWS_S3_ENDPOINT"
echo "Bucket  : Comp_sales_immo"
echo "Fichier : $LOCAL_FILE"
echo "Clé     : $DEST_KEY"

# sanity minimal
python3 - <<PY
import os, boto3
s3=boto3.client("s3", endpoint_url=os.environ["AWS_S3_ENDPOINT"],
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                region_name=os.environ.get("AWS_DEFAULT_REGION","us-east-1"))
print("Sanity OK:", "KeyCount" in s3.list_objects_v2(Bucket=os.environ["DAGSHUB_BUCKET"], MaxKeys=1))
PY

# single-part d'abord
python3 tools/upload_s3_resilient.py "$LOCAL_FILE" "$DAGSHUB_BUCKET" "$DEST_KEY" \
  --endpoint-url "$AWS_S3_ENDPOINT" --path-style --force-single --verbose

# multipart doux (optionnel)
python3 tools/upload_s3_resilient.py "$LOCAL_FILE" "$DAGSHUB_BUCKET" "$DEST_KEY" \
  --endpoint-url "$AWS_S3_ENDPOINT" --path-style --chunk-size-mb 8 --multipart-threshold-mb 16 \
  --max-concurrency 2 --verbose

