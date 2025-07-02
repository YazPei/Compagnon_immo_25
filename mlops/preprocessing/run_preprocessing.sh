#!/bin/bash

set -e

echo "🧼 Lancement du preprocessing"
python src/preprocessing.py \
  --input-path data/clean/df_sales_clean_polars.csv \
  --output-path data/processed/train_clean.csv
