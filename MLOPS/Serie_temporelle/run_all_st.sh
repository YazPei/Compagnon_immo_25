#!/bin/bash

set -e

echo "⚙️ Étape 1 : Split temporel"
docker compose run split

echo "🌀 Étape 2 : Décomposition saisonnière"
docker compose run decompose

echo "📈 Étape 3 : Entraînement SARIMAX"
docker compose run train_sarimax

echo "📊 Étape 4 : Évaluation"
docker compose run evaluate

echo "✅ Pipeline terminée avec succès."
