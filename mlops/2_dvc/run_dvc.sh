#!/bin/bash

set -euo pipefail

# === 📦 Chargement des variables d’environnement ===
ENV_FILE=".env.yaz"

if [ -f "$ENV_FILE" ]; then
    echo "📦 Chargement des variables depuis $ENV_FILE"
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
else
    echo "❌ Fichier $ENV_FILE introuvable. Abandon."
    exit 1
fi

echo "🔍 Vérification des variables : DVC_USER='${DVC_USER}', DVC_TOKEN='(masqué)'"
echo "ST_SUFFIX=${ST_SUFFIX:-undefined}"
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-not set}"

# === 🔐 Configuration du remote DagsHub (non bloquant si déjà existant) ===
echo "🔗 Configuration du remote DVC..."
dvc remote add origin "https://dagshub.com/${DVC_USER}/compagnon_immo.dvc.git" 2>/dev/null || echo "✅ Remote 'origin' déjà présent."
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DVC_USER"
dvc remote modify origin --local password "$DVC_TOKEN"
dvc remote default origin

# === 💾 Mise à jour dynamique de params.yaml avec ST_SUFFIX ===
echo "💾 Écriture de params.yaml avec ST_SUFFIX='$ST_SUFFIX'"
echo "ST_SUFFIX: $ST_SUFFIX" > params.yaml

# === 🚀 Exécution du pipeline DVC ===
echo "📥 Pull des données depuis DagsHub..."
dvc pull --force

echo "🚀 Reproduction du pipeline DVC..."
dvc repro

echo "📥 Import des données dans MLflow..."
python mlops/1_import_donnees/import_data.py --folder-path data --output-folder data

echo "📊 Affichage des métriques..."
dvc metrics show

echo "📈 Affichage des graphiques..."
dvc plots show --html > plots.html

echo "☁️ Push des artefacts vers DagsHub..."
dvc push

echo "✅ Pipeline DVC exécuté avec succès !"

