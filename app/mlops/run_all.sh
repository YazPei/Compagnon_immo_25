#!/bin/bash

set -euo pipefail

# Vérification des dépendances
command -v python >/dev/null 2>&1 || { echo "❌ Python n'est pas installé. Abandon."; exit 1; }
command -v dvc >/dev/null 2>&1 || { echo "❌ DVC n'est pas installé. Abandon."; exit 1; }

# Variables globales
INPUT_PATH="${INPUT_PATH:-data/raw_data.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-data/processed}"

# Exécution du pipeline
echo "🔁 Lancement du pipeline..."
dvc repro

echo "✅ Pipeline exécuté avec succès."