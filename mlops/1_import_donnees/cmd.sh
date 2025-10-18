# &; ENV DAGSHUB 

# 1) Charger tes variables DagsHub 
source /home/vboxuser/Compagnon_new/.dagshub.env

# 2) Test direct Étape 4 (single-part)
python3 tools/upload_s3_resilient.py /home/vboxuser/Compagnon_new/Compagnon_immo_25/merged_sales_data.csv "$DAGSHUB_BUCKET" path/in/bucket/merged_sales_data.csv \
  --endpoint-url "$AWS_S3_ENDPOINT" --path-style --force-single --verbose

# 3) Si OK → Étape 5 (multipart “doux”)
python3 tools/upload_s3_resilient.py /home/vboxuser/Compagnon_new/Compagnon_immo_25/merged_sales_data.csv "$DAGSHUB_BUCKET" path/in/bucket/merged_sales_data.csv \
  --endpoint-url "$AWS_S3_ENDPOINT" --path-style --chunk-size-mb 8 --multipart-threshold-mb 16 \
  --max-concurrency 2 --verbose

# (Option) utiliser le script automatisé
chmod +x /home/vboxuser/Compagnon_new/Compagnon_immo_25/mlops/1_import_donnees/upload_dagshub.sh
/home/vboxuser/Compagnon_new/Compagnon_immo_25/mlops/1_import_donnees/upload_dagshub.sh /home/vboxuser/Compagnon_new/Compagnon_immo_25/merged_sales_data.csv path/in/bucket/merged_sales_data.csv

