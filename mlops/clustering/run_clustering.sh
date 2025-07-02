#!/bin/bash

set -e

echo "📊 Lancement du clustering des données immobilières..."

python mlops/clustering/Clustering.py \
  --input-path data/processed/train_clean.csv \
  --output-path data/interim/df_sales_clustered.csv

echo "✅ Clustering terminé avec succès !"

