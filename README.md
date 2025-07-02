# 🏠 **Pipeline MLOps - Prédiction des prix immobiliers**

Ce dépôt contient une pipeline complète de prédiction des prix immobiliers, divisée en deux approches :

- 🔁 Modélisation par régression (`LightGBM`, `XGBoost`)
- ⏳ Modélisation par séries temporelles (`SARIMAX`)
- 🖥️ App via une API et Streamlit

## 🧩 Stack technologique

- 🐍 `Click` pour l’interface CLI
- 📈 `MLflow` pour le tracking des métriques & modèles
- 📦 `DVC` pour le versionnement des données et étapes
- 🐳 `Docker` pour l’isolation et la reproductibilité
- 🧪 `Makefile` pour l'orchestration simplifiée

---

## 📁 Arborescence


```bash
.
├── api_test
├── mlops/
│   ├── regression/
│   │   ├── regression_pipeline.py
│   │   ├── encoding.py
│   │   ├── train_lgbm.py
│   │   ├── train_xgb.py
│   │   ├── analyse.py
│   │   ├── utils.py
│   │   ├── run_all.sh
│   └── time_series/
│       ├── load_split.py
│       ├── seasonal_decomp.py
│       ├── sarimax_train.py
│       ├── metrics.py
│       ├── utils.py
│       ├── run_all_st.sh

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
├── run_all_full.sh
├── requirements.txt
├── README.md  ⬅️ (ce fichier)
├── Makefile
```


---

## ⚙️ Installation & Lancement (mode local)

```bash
# Installer l’environnement
make install

# Lancer tous les pipelines
make full

# Lancer l'interface MLflow
mlflow ui --port 5001

```

## 🐳 Lancement avec Docker
```bash
# Construire l’image Docker et Exécuter tous les pipelines dans Docker
make docker_auto
```

Nous pouvons également, si on le souhaite ne lancer que le modèle de régression ou le modèle Serie Temporelle:
```bash
make docker_build # Construire l’image Docker
make docker_run_regression # Exécuter uniquement la régression
make docker_run_series # Exécuter uniquement les séries temporelles
```
## 🔁 MLflow Tracking

    Accès : http://localhost:5001
    Tracking automatique de tous les modèles (régression & ST)
    Métriques : RMSE, MAE, R², etc.
    Artéfacts : modèles .joblib, graphes, snapshots encodés

## 🧪 Tests, Export et noettoyage

    Modèles exportés dans exports/
    Snapshots encodés dans X_train.csv, X_test.csv, etc.
    Nettoyage:
```bash
make clean_all
```
Nous pouvons également faire un nettoyage sélectif:
```bash
make clean_exports #  nettoyer les fichiers générés
make clean_dvc # ou nettoyer les DVC cahe
```

