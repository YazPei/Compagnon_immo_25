#!/bin/bash
export PATH=".venv/bin:$PATH"

set -e

echo "🔁 Lancement du pipeline des Séries Temporelles avec DVC"
dvc repro evaluate

echo "📈 Visualisation du graphe"
dvc dag

#echo "⚙️ Étape 1 : Split temporel"
#docker compose run split

#echo "🌀 Étape 2 : Décomposition saisonnière"
#docker compose run decompose

#echo "📈 Étape 3 : Entraînement SARIMAX"
#docker compose run train_sarimax

#echo "📊 Étape 4 : Évaluation"
#docker compose run evaluate

echo "✅ Pipeline Serie Temporelle terminée avec succès."


