# 🏡 MLOps - Modélisation Régressive Immobilière

Ce projet exécute un pipeline complet de modélisation des prix immobiliers au m², avec :

- 💡 Prétraitement & encodage (`encoding.py`)
- 🧠 Modèles LGBM ou XGBoost (avec Optuna)
- 🔍 Analyse SHAP & résidus
- 🔄 Suivi complet via **MLflow** & **DVC**
- 🐳 Dockerisation de chaque étape
- 🔁 Exécutable en CLI via Click

---

## 🚀 Arborescence du projet

```bash
.
├── docker-compose.yml
├── Dockerfile.encoding
├── Dockerfile.lgbm
├── Dockerfile.xgb
├── Dockerfile.analyse
├── dvc.yaml
├── requirements.txt
├── run_all.sh
├── src/
│   ├── regression_pipeline.py     ← CLI central
│   ├── encoding.py                ← Encodage & split
│   ├── train_lgbm.py              ← Modèle LGBM + Optuna
│   ├── train_xgb.py               ← Modèle XGBoost
│   ├── analyse.py                 ← SHAP + résidus
│   └── utils.py                   ← Métriques & SHAP utils
└── mlops/
    ├── data/df_cluster.csv        ← Données brutes
    ├── encoded/                   ← Données encodées + modèles
```
## Etapes
### 1. Execution bash :
```bash
chmod +x run_all.sh
./run_all.sh
```

### 2. Docker
Execution via docker-compose 
```bash
docker-compose build
docker-compose run encode
docker-compose run train_lgbm
docker-compose run analyse
```

### 3. MLflow Tracking

MLflow logue automatiquement :
    Hyperparamètres
    Scores (MSE, RMSE, MAE, R²)
    Artéfacts (modèle joblib, features…)

Lancez l’interface :
```bash
mlflow ui
```
Puis ouvrez http://localhost:5000

### 4. DVC
Rejouez le pipeline entier avec :
```bash
dvc repro
```
