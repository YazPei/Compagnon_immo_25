#!/usr/bin/env bash
set -euo pipefail

# UID/GID de l'utilisateur Airflow dans le conteneur
AUID="${AIRFLOW_UID:-50000}"
AGID="${AIRFLOW_GID:-0}"           # groupe 0 = root pour compatibilité
AF_HOME="/opt/airflow"

# Dossiers montés en volumes
DIRS=(
  "$AF_HOME/logs"
  "$AF_HOME/dags"
  "$AF_HOME/repo"
  "$AF_HOME/data"
  "$AF_HOME/exports"
  "$AF_HOME/mlops"
)

echo "[perm] ensure directories exist"
for d in "${DIRS[@]}"; do
  mkdir -p "$d"
done

echo "[perm] chown recursively to ${AUID}:${AGID}"
for d in "${DIRS[@]}"; do
  chown -R "${AUID}:${AGID}" "$d" || true
  chmod -R g+rwX "$d" || true
done

echo "[perm] done"

