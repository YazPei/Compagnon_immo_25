#!/usr/bin/env bash

set -e

echo "=== Configuration de l'environnement Airflow/DagsHub ==="

read -p "Ton login DagsHub: " dagshub_user
read -p "Owner/organisation du repo: " repo_owner
read -p "Nom du repo: " repo_name
read -s -p "Ton token DagsHub (copie/colle ici, caché): " dagshub_token
echo

cat > .env <<EOF
DAGSHUB_USER=${dagshub_user}
DAGSHUB_TOKEN=${dagshub_token}
DAGSHUB_REPO_OWNER=${repo_owner}
DAGSHUB_REPO_NAME=${repo_name}

MLFLOW_TRACKING_URI=https://dagshub.com/${repo_owner}/${repo_name}.mlflow
MLFLOW_TRACKING_USERNAME=${dagshub_user}
MLFLOW_TRACKING_PASSWORD=${dagshub_token}
EOF

echo "✅ Fichier .env généré ! Tu peux maintenant lancer : docker compose up -d"
