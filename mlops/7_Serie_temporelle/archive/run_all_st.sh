#!/bin/bash
export PATH=".venv/bin:$PATH"
set -e

echo "🔁 Lancement du pipeline des Séries Temporelles avec Docker + DVC"

# Date du jour pour suffixer les fichiers
TODAY=$(date +%Y%m%d)
export ST_SUFFIX="_${TODAY}"

echo "⚙️ Étape 1 : Split temporel avec suffixe $ST_SUFFIX"
docker compose run --rm split \
  --input-path /app/data/df_sales_clean_ST.csv \
  --taux-path /app/data/taux_immo.xlsx \
  --geo-path /app/data/contours-codes-postaux.geojson \
  --output-folder /app/exports/st \
  --suffix "$ST_SUFFIX"

echo "🌀 Étape 2 : Décomposition saisonnière"
docker compose run --rm decompose \
  --input-folder /app/exports/st \
  --output-folder /app/exports/st \
  --suffix "$ST_SUFFIX"
  

echo "📈 Étape 3 : Entraînement SARIMAX"
docker compose run --rm train_sarimax \
  --input-folder /app/exports/st \
  --output-folder /app/exports/st \
  --suffix "$ST_SUFFIX"


echo "📊 Étape 4 : Évaluation"
docker compose run --rm evaluate \
  --input-folder data/split/ \
  --output-folder outputs/forecast_eval/ \
  --model-folder outputs/best/
  --suffix "$ST_SUFFIX"


echo "📈 Visualisation du graphe DVC"
dvc dag

echo "✅ Pipeline Série Temporelle terminée avec succès."

