#!/bin/bash

set -e  # Arrêter le script en cas d'erreur
set -o pipefail  # Propager les erreurs dans les pipes

# Vérification des dépendances
echo "🔍 Vérification des dépendances..."

command -v python >/dev/null 2>&1 || { echo "❌ Python n'est pas installé. Abandon."; exit 1; }
command -v mlflow >/dev/null 2>&1 || { echo "❌ MLflow n'est pas installé. Abandon."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker n'est pas installé. Abandon."; exit 1; }

echo "✅ Toutes les dépendances sont installées."

# Variables globales
INPUT_PATH="data/raw_data.csv"
OUTPUT_PATH="data/processed"
LOG_FILE="logs/run_all.log"

# Création des dossiers nécessaires
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$OUTPUT_PATH"

# Exécution des étapes du pipeline
echo "🚀 Lancement du pipeline de prétraitement..."
python app/services/preprocessing/preprocessing.py --input-path "$INPUT_PATH" --output-path "$OUTPUT_PATH" | tee -a "$LOG_FILE"

echo "✅ Pipeline terminé avec succès."