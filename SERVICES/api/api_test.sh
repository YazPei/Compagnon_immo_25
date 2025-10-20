#!/bin/bash
set -euo pipefail

echo "🐳 Tests de l'API avec Docker..."
echo "🚀 Démarrage de l'environnement de test complet..."
docker compose --profile test up --build --abort-on-container-exit --exit-code-from api-test --quiet-pull
echo "🛑 Nettoyage de l'environnement de test..."
docker compose --profile test down -v
