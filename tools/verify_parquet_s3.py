# path: tools/verify_parquet_s3.py
#!/usr/bin/env python3
"""
Vérifie un Parquet sur S3 (DagsHub S3-compatible):
- Lit schéma + 5 premières lignes.
- Utilise env: AWS_S3_ENDPOINT, AWS_DEFAULT_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DAGSHUB_BUCKET
- Clé à vérifier via env VERIFY_KEY ou argument --key
"""
from __future__ import annotations
import os, sys, io, argparse
import boto3
from botocore.config import Config
import pyarrow.parquet as pq

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.getenv("VERIFY_KEY"), help="Chemin S3 de l'objet Parquet")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    if not args.key:
        print("ERROR: --key / $VERIFY_KEY requis", file=sys.stderr); return 2

    endpoint = os.getenv("AWS_S3_ENDPOINT") or None
    region   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    bucket   = os.getenv("DAGSHUB_BUCKET")
    ak       = os.getenv("AWS_ACCESS_KEY_ID")
    sk       = os.getenv("AWS_SECRET_ACCESS_KEY")
    st       = os.getenv("AWS_SESSION_TOKEN")

    if not bucket or not ak or not sk:
        print("ERROR: variables AWS/DAGSHUB manquantes", file=sys.stderr); return 2

    cfg = Config(region_name=region, retries={"max_attempts": 5, "mode": "standard"}, s3={"addressing_style":"path"})
    s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=ak, aws_secret_access_key=sk, aws_session_token=st, config=cfg)

    # Télécharge en mémoire (ok pour fichiers raisonnables); sinon, utiliser un fichier temporaire
    obj = s3.get_object(Bucket=bucket, Key=args.key)
    buf = io.BytesIO(obj["Body"].read())

    pqf = pq.ParquetFile(buf)
    schema = pqf.schema_arrow
    print("=== SCHEMA ===")
    for f in schema:
        print(f"-", f)

    print("=== HEAD(5) ===")
    table = pqf.read_row_groups([0]) if pqf.num_row_groups >= 1 else pqf.read()
    df = table.to_pandas().head(5)
    with io.StringIO() as s:
        df.to_string(buf=s, max_cols=30, max_rows=5)
        print(s.getvalue())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

