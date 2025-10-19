# 0) Env
set -e
set -a; . ~/.dagshub.env; set +a
: "${DAGSHUB_USER:?missing}"; : "${DAGSHUB_TOKEN:?missing}"

# 1) Re-upload CSV ORIGINAL (garde les ;)
python3 tools/dagshub_s3.py upload \
  --file merged_sales_data.csv \
  --key  data/merged_sales_data.csv \
  --content-type "text/csv" \
  --force-single --verbose

# 2) OPTION: copie "viewer-friendly" TSV (affichage tabulaire sans config)
python3 tools/dagshub_s3.py upload \
  --file merged_sales_data.csv \
  --key  data/merged_sales_data.tsv \
  --normalize-csv --to tsv --encoding utf-8 \
  --force-single --verbose

# 3) Recréer/maj le DATASET + Source S3 + Build (delimiter=';')
#    (utilise ton script tools/dagshub_dataset_upsert.py si tu l’as)
python3 tools/dagshub_dataset_upsert.py \
  --owner "YazPei" \
  --repo  "Compagnon_immo_25" \
  --dataset "merged_sales_data" \
  --prefix "data/" \
  --include "**/*.csv" \
  --delimiter ";" \
  --encoding "utf-8" \
  --has-header

# 4) Vérif rapide côté S3
python3 tools/dagshub_s3.py list --prefix "data/"
python3 tools/dagshub_s3.py cat  --key "data/merged_sales_data.csv" --lines 5

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
dvc import-url --to-remote "s3://$DAGSHUB_BUCKET/data/merged_sales_data.csv" data/merged_sales_data_dvc.csv || true
git add data/merged_sales_data_dvc.csv.dvc .dvc/config
git commit -m "restore dataset tracking" || true

