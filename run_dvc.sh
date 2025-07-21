#!/bin/bash
echo "ST_SUFFIX=${ST_SUFFIX}"
set -e  # Stop on first error

echo "📦 Activation de l'environnement..."

# À exécuter une fois :
dvc remote add origin https://dagshub.com/yazpei/compagnon_immo.dvc
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
