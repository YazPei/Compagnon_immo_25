# 0) Env
#!/usr/bin/env bash
set -euo pipefail

# --- ENV ---
set -a
[ -f ~/.dagshub.env ] && . ~/.dagshub.env || true
[ -f .env ] && . .env || true
set +a
: "${DAGSHUB_USER:?missing}"; : "${DAGSHUB_TOKEN:?missing}"

# --- Params
FILE="${1:-DVF_donnees_macroeco.csv}"              # 1er arg = fichier local
KEY="${2:-data/DVF_donnees_macroeco.csv}"          # 2e arg = clé S3
MAKE_TSV="${MAKE_TSV:-0}"                       # export tsv optionnel (0/1)

# 1) Re-upload CSV ORIGINAL 
# --- Upload CSV “as-is” ---
python3 tools/dagshub_s3.py upload \
  --file "${FILE}" \
  --key  "${KEY}" \
  --content-type "text/csv" \
  --force-single --verbose
  
  
if [[ "${MAKE_TSV}" == "1" ]]; then
  python3 tools/dagshub_s3.py upload \
    --file "${FILE}" \
    --key  "${KEY%.*}.tsv" \
    --normalize-csv --to tsv --encoding utf-8 \
    --force-single --verbose
fi

# 3) Recréer/maj le DATASET + Source S3 + Build (delimiter=';')
#    (utilise ton script tools/dagshub_dataset_upsert.py si tu l’as)
python3 tools/dagshub_dataset_upsert.py \
  --owner "YazPei" \
  --repo  "Compagnon_immo_25" \
  --dataset "DVF_donnees_macroeco" \
  --include "**/*.csv" \
  --delimiter ";" \
  --encoding "utf-8" \
  --has-header

# 4) Vérif rapide côté S3
python3 tools/dagshub_s3.py list 
python3 tools/dagshub_s3.py cat  --key "DVF_donnees_macroeco.csv" --lines 5

# 5) (Option) si tu avais supprimé le tracking DVC du fichier:
dvc init -q || true
dvc remote remove dagshub-s3 2>/dev/null || true
dvc remote add -f dagshub-s3 "s3://$DAGSHUB_BUCKET"
dvc remote modify dagshub-s3 endpointurl "$AWS_S3_ENDPOINT"
dvc remote modify dagshub-s3 region "$AWS_DEFAULT_REGION"
dvc remote modify --local dagshub-s3 access_key_id "$AWS_ACCESS_KEY_ID"
dvc remote modify --local dagshub-s3 secret_access_key "$AWS_SECRET_ACCESS_KEY"
[ -n "$AWS_SESSION_TOKEN" ] && dvc remote modify --local dagshub-s3 session_token "$AWS_SESSION_TOKEN" || true

# Pointer le CSV S3 sans le re-pousser
dvc import-url --to-remote "s3://$DAGSHUB_BUCKET/DVF_donnees_macroeco.csv" data/DVF_donnees_macroeco.csv || true
git add data/DVF_donnees_macroeco.csv.dvc .dvc/config
git commit -m "restore dataset tracking" || true

