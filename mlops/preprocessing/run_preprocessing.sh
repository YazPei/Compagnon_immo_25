#!/bin/bash

set -e

echo "🧼 Lancement du preprocessing"
python preprocessing.py \
  --input-path data/df_sales_clean_polars.csv \
  --output-path data/train_clean.csv \
  --output-path data/test_clean.csv
