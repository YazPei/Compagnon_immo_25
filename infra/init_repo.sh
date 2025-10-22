#!/usr/bin/env bash
set -euo pipefail

REPO=/opt/airflow/repo
cd "$REPO"

echo "[init] git safe.directory"
git config --global --add safe.directory "$REPO" || true

echo "[init] create venv at $REPO/.venv"
python -m venv "$REPO/.venv"
source "$REPO/.venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# DVC remote (optionnel) - adapte si tu en as besoin
# dvc remote modify <name> --local access_key_id ...
# dvc remote modify <name> --local secret_access_key ...

echo "[init] airflow variables"
if [ -f "$REPO/infra/variables.json" ]; then
  airflow variables import "$REPO/infra/variables.json" || true
fi

echo "[init] done"

