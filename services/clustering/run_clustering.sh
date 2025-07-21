#!/bin/bash

set -e

echo "📊 Lancement du clustering des données immobilières..."

python mlops/clustering/Clustering.py \
  --input-path data/train_clean.csv \
  --output-path data/"df_cluster.csv".csv \
  --output-path data/"df_sales_clean_ST.csv".csv

echo "✅ Clustering terminé avec succès !"

