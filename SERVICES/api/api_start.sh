#!/bin/bash
set -euo pipefail

echo "🚀 Démarrage de l'API..."
docker compose up api --build -d
