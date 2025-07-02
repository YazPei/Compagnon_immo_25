Ce dépôt contient une pipeline complète de prédiction des prix immobiliers, divisée en deux approches :

    🔁 Modélisation par régression (LightGBM/XGBoost)

    ⏳ Modélisation par séries temporelles (SARIMAX)

Le tout est orchestré en pipelines modulaires avec :

    Click pour l’interface CLI

    MLflow pour le tracking des métriques & modèles

    DVC pour le versionnement des données et étapes

    Docker pour l’isolation et la reproductibilité

📁 Arborescence

.
├── mlops/
│   ├── regression/
│   │   ├── regression_pipeline.py
│   │   ├── encoding.py
│   │   ├── train_lgbm.py
│   │   ├── train_xgb.py
│   │   ├── analyse.py
│   │   ├── utils.py
│   └── time_series/
│       ├── load_split.py
│       ├── seasonal_decomp.py
│       ├── sarimax_train.py
│       ├── metrics.py
│       ├── utils.py
├── data/
│   ├── df_cluster.csv
│   └── df_sales_clean_ST.csv
├── exports/
│   ├── reg/
│   │   ├── X_train.csv, X_test.csv, y_train.csv, y_test.csv
│   │   ├── lightgbm_model.joblib, xgboost_model.joblib
│   └── st/
│       ├── X_train.csv, X_test.csv, y_train.csv, y_test.csv
│       └── model_sarimax_cluster_*.pkl
├── Dockerfile.*
├── docker-compose.yml
├── dvc.yaml
├── run_all.sh
├── requirements.txt
├── README_series_temporality.md
├── README.md  ⬅️ (ce fichier)

⚙️ Environnement

# Installer les dépendances
pip install -r requirements.txt

# Lancer MLflow localement
mlflow ui --port 5001

🏗️ Lancement pipeline Régression
Étapes

# Encodage
python mlops/regression/regression_pipeline.py encode --data-path ./data/df_cluster.csv --output ./exports/reg

# Entraînement LGBM
python mlops/regression/regression_pipeline.py train-lgbm --encoded-folder ./exports/reg

# Entraînement XGBoost
python mlops/regression/regression_pipeline.py train-xgb --encoded-folder ./exports/reg --use-gpu

# Analyse
python mlops/regression/regression_pipeline.py analyse --encoded-folder ./exports/reg --model lightgbm

⏳ Lancement pipeline Série Temporelle
En un seul script

bash run_all.sh

Étape par étape

docker compose run split
docker compose run decompose
docker compose run train_sarimax
docker compose run evaluate

🔁 MLflow Tracking

    Accès : http://localhost:5001
    Tracking automatique de tous les modèles (régression & ST)
    Métriques : RMSE, MAE, R², etc.
    Artéfacts : modèles .joblib, graphes, snapshots encodés

🧪 Tests & Export

    Modèles exportés dans exports/
    Snapshots encodés dans X_train.csv, X_test.csv, etc.
