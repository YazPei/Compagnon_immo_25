# Autorise '>' comme préfixe de recette (au lieu du TAB)
.RECIPEPREFIX := >

PY ?= python3
MAKE ?= make

.PHONY: s3_sanity_dagshub import_s3_dagshub s3_import_data

s3_sanity_dagshub: ## Vérifie l'accès à l'objet DagsHub (HEAD + list)
> @[ -n "$(S3_ENDPOINT)" ] || (echo "Set S3_ENDPOINT (ex: https://dagshub.com/api/v1/repo-buckets/s3/YazPei)"; exit 2)
> @[ -n "$(BUCKET)" ]     || (echo "Set BUCKET (ex: Compagnon_immo_25)"; exit 2)
> @[ -n "$(KEY)" ]        || (echo "Set KEY (ex: merged_sales_data.csv)"; exit 2)
> @[ -n "$$AWS_ACCESS_KEY_ID" ]     || (echo "Set AWS_ACCESS_KEY_ID=YazPei"; exit 2)
> @[ -n "$$AWS_SECRET_ACCESS_KEY" ] || (echo "Set AWS_SECRET_ACCESS_KEY=<PAT DagsHub>"; exit 2)
> $(PY) - <<'PY'
import os, boto3
from botocore.config import Config
s3=boto3.client("s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("REGION","us-east-1"),
    config=Config(signature_version="s3v4", s3={"addressing_style":"path"}))
b=os.environ["BUCKET"]; k=os.environ["KEY"]
print("Head:", s3.head_object(Bucket=b, Key=k))
resp=s3.list_objects_v2(Bucket=b, MaxKeys=20)
print("Keys:", [o["Key"] for o in (resp.get("Contents") or [])])
PY

import_s3_dagshub: ## Import S3 (DagsHub, auth requise)
> @[ -n "$(S3_ENDPOINT)" ] || (echo "Set S3_ENDPOINT"; exit 2)
> @[ -n "$(BUCKET)" ]     || (echo "Set BUCKET"; exit 2)
> @[ -n "$(KEY)" ]        || (echo "Set KEY"; exit 2)
> @[ -n "$(REGION)" ]     || (echo "Set REGION (us-east-1)"; exit 2)
> @[ -n "$$AWS_ACCESS_KEY_ID" ]     || (echo "Set AWS_ACCESS_KEY_ID=YazPei"; exit 2)
> @[ -n "$$AWS_SECRET_ACCESS_KEY" ] || (echo "Set AWS_SECRET_ACCESS_KEY=<PAT DagsHub>"; exit 2)
> mkdir -p data/state data/incremental
> PYTHONIOENCODING=UTF-8 LC_ALL=C.UTF-8 LANG=C.UTF-8 \
> $(PY) mlops/1_import_donnees/import_data.py \
>   --source-mode s3 \
>   --s3-endpoint-url "$(S3_ENDPOINT)" \
>   --s3-bucket "$(BUCKET)" \
>   --s3-key "$(KEY)" \
>   --s3-region "$(REGION)" \
>   --output-folder data/incremental \
>   --cumulative-path data/df_sample.csv \
>   --checkpoint-path data/state/checkpoint.parquet \
>   --date-column "$(DATE_COL)" \
>   --key-columns "$(KEY_COLS)" \
>   --sep ";"

s3_import_data: ## Import DagsHub avec defaults
> $(MAKE) import_s3_dagshub \
>   S3_ENDPOINT=https://dagshub.com/api/v1/repo-buckets/s3/YazPei \
>   BUCKET=Compagnon_immo_25 \
>   KEY=merged_sales_data.csv \
>   REGION=us-east-1 \
>   DATE_COL=date \
>   KEY_COLS=idannonce
