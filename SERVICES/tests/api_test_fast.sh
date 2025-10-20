#!/bin/bash
set -euo pipefail

echo "⚡ Tests de l'API rapides avec Docker..."
echo "🚀 Démarrage de l'environnement de test (utilise les images existantes)..."
docker compose --profile test up --abort-on-container-exit --exit-code-from api-test --quiet-pull
echo "🛑 Nettoyage de l'environnement de test..."
docker compose --profile test down -v
