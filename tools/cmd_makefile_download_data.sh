# 0) charger les variables DagsHub
set -a; . ~/.dagshub.env; set +a

# 1) télécharger le CSV
make -f Makefile.s3 s3-download S3_KEY="data/merged_sales_data.csv" S3_OUT="downloads/"


# 2) aperçu rapide (10 lignes)
make -f Makefile.s3 s3-cat S3_KEY="data/merged_sales_data.csv"

