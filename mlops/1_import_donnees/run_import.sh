#!/usr/bin/env bash
set -euo pipefail

# 0) CWD = racine du repo
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# 1) Charger .env (normalisé)
ENV_PATH="$ROOT/.env"
if [ ! -f "$ENV_PATH" ]; then
  echo "ERR: .env introuvable: $ENV_PATH" >&2
  exit 1
fi
sed -i 's/\r$//' "$ENV_PATH"
sed -i '1s/^\xEF\xBB\xBF//' "$ENV_PATH"
awk 'NF && $0 !~ /=/ {print "ERR .env:", NR ":" $0; bad=1} END{exit bad}' "$ENV_PATH"
set -a; . "$ENV_PATH"; set +a

# 2) Choisir un Python existant
if [ -n "${PY:-}" ] && [ -x "$PY" ]; then :
elif [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python
elif [ -x ./.s3venv/bin/python ]; then PY=./.s3venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY="$(command -v python3)"
else echo "ERR: Python introuvable (./.venv/bin/python ni python3)"; exit 127; fi
echo "[info] Using PY=$PY"

# 3) Modules requis
"$PY" - <<'PY'
import importlib, sys
for mod in ("boto3","s3fs","pandas"):
    try: importlib.import_module(mod)
    except Exception as e:
        print(f"[err] Python module missing: {mod} -> {e}"); sys.exit(127)
print("[ok] required modules present")
PY

# 4) Env requis pour S3
for k in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION AWS_ENDPOINT_URL_S3 BUCKET KEY; do
  [ -n "${!k:-}" ] || { echo "ERR: $k manquant"; exit 2; }
done

# 5) Lancer l'import
exec "$PY" mlops/1_import_donnees/import_data.py \
  --source-mode s3 \
  --s3-endpoint-url "${AWS_ENDPOINT_URL_S3}" \
  --s3-bucket "${BUCKET}" \
  --s3-key "${KEY}" \
  --s3-region "${AWS_DEFAULT_REGION:-us-east-1}" \
  --output-folder data/incremental \
  --cumulative-path data/df_sample.csv \
  --checkpoint-path data/state/checkpoint.parquet \
  --date-column "${DATE_COL:-date}" \
  --key-columns "${KEY_COLS:-idannonce}" \
  --sep ';'
