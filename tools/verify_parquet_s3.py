import os, sys, tempfile
import boto3
import pyarrow.parquet as pq
import pandas as pd
from botocore.config import Config

def main():
    key = os.environ.get("VERIFY_KEY") or (len(sys.argv) > 1 and sys.argv[1])
    if not key:
        print("Usage: VERIFY_KEY=<s3_key> tools/verify_parquet_s3.py", file=sys.stderr)
        sys.exit(2)

    endpoint = os.environ["AWS_S3_ENDPOINT"]
    region   = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    bucket   = os.environ["DAGSHUB_BUCKET"]
    ak = os.environ["AWS_ACCESS_KEY_ID"]
    sk = os.environ["AWS_SECRET_ACCESS_KEY"]

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=Config(s3={"addressing_style":"path"}, retries={"max_attempts": 8, "mode":"adaptive"}),
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tmp = tf.name
    s3.download_file(bucket, key, tmp)

    # Schéma
    pf = pq.ParquetFile(tmp)
    print("=== Parquet schema ===")
    print(pf.schema)
    # Head 5
    df = pd.read_parquet(tmp)
    print("\n=== Head (5) ===")
    print(df.head(5))
    print(f"\nRows: {len(df):,}  Cols: {len(df.columns)}  Key: s3://{bucket}/{key}")

if __name__ == "__main__":
    main()
PY
chmod +x tools/verify_parquet_s3.py

# --- Makefile.s3 (APPEND ces cibles ; RECIPEPREFIX est déjà défini dans ton fichier) ---
cat >> Makefile.s3 <<'EOF'

# ===== Parquet defaults =====
PARQ_LOCAL ?= data/merged_sales_data.parquet
CSV_LOCAL  ?= merged_sales_data.csv
PARQ_KEY   ?= $(PARQ_LOCAL)
CSV_SEP    ?= ;

# s3-install déjà présent: on s'assure d'avoir les deps nécessaires
s3-install: s3-venv
> $(S3_PIP) -q install boto3 botocore pandas click mlflow pyarrow fastparquet

# Convert CSV(;) -> Parquet (streaming)
.PHONY: s3-csv2parquet-stream
s3-csv2parquet-stream: s3-install
> $(S3_PY) tools/csv_to_parquet.py --src "$(CSV_LOCAL)" --dst "$(PARQ_LOCAL)" --sep "$(CSV_SEP)"

# Upload le Parquet sous data/
.PHONY: s3-upload-parquet
s3-upload-parquet: s3-env
> set -a; source $$HOME/.dagshub.env; set +a; \
> $(S3_PY) tools/dagshub_s3.py upload --file "$(PARQ_LOCAL)" --key "$(PARQ_KEY)" --force-single --verbose

# Vérifie le Parquet directement sur S3 (schéma + 5 lignes)
.PHONY: s3-verify-parquet
s3-verify-parquet: s3-env
> set -a; source $$HOME/.dagshub.env; set +a; \
> VERIFY_KEY="$(PARQ_KEY)" $(S3_PY) tools/verify_parquet_s3.py

# Branche s3-import pour lire le Parquet par défaut
IMP_OUT_DIR    ?= data/incremental
IMP_CUMUL_PATH ?= data/df_sample.csv
IMP_CHECKPOINT ?= data/checkpoint.parquet
IMP_DATE_COL   ?=
IMP_KEY_COLS   ?=
IMP_SEP        ?= ;
# IMPORTANT: Par défaut on lit le Parquet uploadé
IMP_S3_KEY     ?= $(PARQ_KEY)

.PHONY: s3-import
s3-import: s3-env
> set -a; source $$HOME/.dagshub.env; set +a; \
> $(S3_PY) /home/vboxuser/Compagnon_new/Compagnon_immo_25/mlops/1_import_donnees/import_data.py \
>   --source-mode s3 \
>   --output-folder "$(IMP_OUT_DIR)" \
>   --cumulative-path "$(IMP_CUMUL_PATH)" \
>   --checkpoint-path "$(IMP_CHECKPOINT)" \
>   --sep "$(IMP_SEP)" \
>   --s3-key "$(IMP_S3_KEY)" \
>   $(if $(IMP_DATE_COL),--date-column "$(IMP_DATE_COL)",) \
>   $(if $(IMP_KEY_COLS),--key-columns "$(IMP_KEY_COLS)",)

# Pipeline parquet complète: convert -> upload parquet -> verify -> import -> list
.PHONY: s3-pipeline-parquet
s3-pipeline-parquet: s3-csv2parquet-stream s3-upload-parquet s3-verify-parquet s3-import s3-list
> @echo "✅ Pipeline Parquet terminée (upload+verify+import). Dans DagsHub, datasource prefix=data/ puis Build/Index."
EOF
