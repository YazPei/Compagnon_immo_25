#!/usr/bin/env bash
set -euo pipefail
# toujours source le .env du repo racine
ENV_PATH="$(git rev-parse --show-toplevel)/.env"
[ -f "$ENV_PATH" ] || { echo "ERR: .env introuvable à $ENV_PATH"; exit 1; }

# hygiène: CRLF + BOM + format
sed -i 's/\r$//' "$ENV_PATH"
sed -i '1s/^\xEF\xBB\xBF//' "$ENV_PATH"
awk 'NF && $0 !~ /=/ {print "ERR .env:", NR ":" $0; bad=1} END{exit bad}' "$ENV_PATH"

set -a; . "$ENV_PATH"; set +a


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

