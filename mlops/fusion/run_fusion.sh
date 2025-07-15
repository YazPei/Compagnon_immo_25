#!/bin/bash

set -e

echo "🌐 Lancement de la fusion des données IPS & géo"
cd ..
dvc pull 

python -m mlops/fusion/fusion_geo_dvf \
  --folder-path1 data \
  --folder-path2 data \
  --output-folder data/clean
  
  


