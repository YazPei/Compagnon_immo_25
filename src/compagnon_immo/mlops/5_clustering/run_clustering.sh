# path: mlops/5_clustering/run_clustering.sh
#!/usr/bin/env bash
set -euo pipefail
# why: exécuter exactement comme dvc.yaml
python mlops/5_clustering/Clustering.py \
  --input-path data/processed \
  --output-path exports/df_cluster.csv
echo "✅ Clustering terminé."

