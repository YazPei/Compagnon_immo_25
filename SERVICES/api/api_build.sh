#!/bin/bash
set -euo pipefail

echo "🔨 Construction de l'image API..."
DOCKER_BUILDKIT=0 docker build -t compagnon_immo-api .
