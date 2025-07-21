#!/bin/bash
set -e
export ST_SUFFIX=_$(date +%Y%m%d)
docker compose run --rm run_full


# Lancement du script Python
echo "🔄 Lancement fusion_geo_dvf.py"
python /app/mlops/fusion/fusion_geo_dvf.py \
  --folder-path1 /app/data \
  --folder-path2 /app/data \
  --output-folder /app/data

