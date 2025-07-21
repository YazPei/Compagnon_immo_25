#!/bin/bash
echo "ST_SUFFIX=${ST_SUFFIX}"

set -euo pipefail
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow:5000}


echo "📥 Import des données dans MLflow..."
python mlops/import_donnees/import_data.py --folder-path data --output-folder data




# Configuration DVC (à faire une seule fois si pas déjà dans .dvc/config)
echo "🔗 Configuration du remote DVC..."
dvc remote add origin https://dagshub.com/yazpei/compagnon_immo.dvc || true
dvc remote modify origin --local auth basic
dvc remote modify origin --local user yazpei
dvc remote modify origin --local password "$DVC_TOKEN"
dvc remote default origin

echo "🚀 Lancement du pipeline DVC..."
dvc repro

echo "📊 Affichage des métriques..."
dvc metrics show

echo "📈 Affichage des graphiques (si plots.yaml)..."
dvc plots show --html > plots.html

echo "☁️ Push des artefacts dans le remote DVC..."
dvc push

echo "✅ Pipeline DVC exécuté avec succès !"

