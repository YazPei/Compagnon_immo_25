set -euo pipefail
OWNER="${1:-YazPei}"
REPO="${2:-Compagnon_immo_25}"
DATASET="${3:-}"
if [[ -z "${DAGSHUB_USER:-}" || -z "${DAGSHUB_TOKEN:-}" ]]; then
  echo "ERROR: export DAGSHUB_USER and DAGSHUB_TOKEN first." >&2; exit 2
fi
if [[ -z "$DATASET" ]]; then
  echo "Usage: DAGSHUB_USER=.. DAGSHUB_TOKEN=.. $0 <owner> <repo> <dataset_name>" >&2
  echo "Datasets existants:" >&2
  curl -s -u "$DAGSHUB_USER:$DAGSHUB_TOKEN" \
    "https://dagshub.com/api/v1/repos/$OWNER/$REPO/datasets" || true
  exit 2
fi
echo "# Trigger build for dataset: $OWNER/$REPO -> $DATASET"
curl -s -X POST -u "$DAGSHUB_USER:$DAGSHUB_TOKEN" \
  "https://dagshub.com/api/v1/repos/$OWNER/$REPO/datasets/$DATASET/build" \
  | sed 's/},/\n/g'
echo "OK"

