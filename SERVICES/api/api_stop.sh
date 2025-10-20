#!/bin/bash
set -euo pipefail

echo "🛑 Arrêt de l'API..."
docker compose down api
