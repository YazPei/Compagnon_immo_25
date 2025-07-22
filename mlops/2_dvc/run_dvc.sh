#!/bin/bash

set -euo pipefail

# Load .env
ENV_FILE=".env.yaz"
echo "🔍 Vérification des variables : DVC_USER='$DVC_USER', DVC_TOKEN='(masqué)'"

if [ -f "$ENV_FILE" ]; then
    echo "📦 Chargement des variables depuis $ENV_FILE"
    set -o allexport
    source "$ENV_FILE"
    set +o allexport

else
    echo "❌ Fichier $ENV_FILE introuvable. Abandon."
    exit 1
fi
echo "🔍 Vérification des variables : DVC_USER='$DVC_USER', DVC_TOKEN='(masqué)'"

echo "ST_SUFFIX=${ST_SUFFIX:-undefined}"
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-not set}"
echo "🔍 Vérification des variables : DVC_USER='$DVC_USER', DVC_TOKEN='(masqué)'"

# Optionnel : vérifie présence des credentials
if [[ -z "${DVC_USER:-}" || -z "${DVC_TOKEN:-}" ]]; then
    echo "❌ Variables DVC_USER ou DVC_TOKEN manquantes dans .env"
    exit 1
fi
echo "🔍 Vérification des variables : DVC_USER='$DVC_USER', DVC_TOKEN='(masqué)'"

export MLFLOW_TRACKING_URI






echo "🔐 Utilisateur DagsHub détecté : $DVC_USER"

# Configuration DVC (à faire une seule fois si pas déjà dans .dvc/config)
echo "🔗 Configuration du remote DVC..."

dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DVC_USER"
dvc remote modify origin --local password "$DVC_TOKEN"
dvc remote default origin


echo "🚀 Lancement du pipeline DVC..."
dvc pull
dvc repro
echo "📥 Import des données dans MLflow..."
python mlops/1_import_donnees/import_data.py --folder-path data --output-folder data

echo "📊 Affichage des métriques..."
dvc metrics show

echo "📈 Affichage des graphiques (si plots.yaml)..."
dvc plots show --html > plots.html

echo "☁️ Push des artefacts dans le remote DVC..."
dvc push

echo "✅ Pipeline DVC exécuté avec succès !"

