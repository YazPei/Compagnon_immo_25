#!/bin/bash

set -euo pipefail

# Chargement des variables d'environnement
export DVC_USER="${DVC_USER:-ci_user}"
export DVC_TOKEN="${DVC_TOKEN:-ci_token}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

echo "🔗 Configuration du remote DVC..."
dvc remote add origin "https://dagshub.com/${DVC_USER}/compagnon_immo_25.dvc.git" 2>/dev/null || echo "✅ Remote 'origin' déjà présent."
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DVC_USER"
dvc remote modify origin --local password "$DVC_TOKEN"

echo "📥 Pull des données depuis DagsHub..."
dvc pull --force

echo "✅ Pipeline DVC exécuté avec succès !"