# 🏠 **Pipeline MLOps - Prédiction des prix immobiliers**

Ce dépôt contient une pipeline complète pour la prédiction des prix immobiliers, divisée en deux approches principales :

- 🔁 **Modélisation par régression** (`LightGBM`, `XGBoost`)
- ⏳ **Modélisation par séries temporelles** (`SARIMAX`)
- 🌐 **API FastAPI** pour servir les modèles et les prédictions

---

## 🧩 **Stack technologique**

- 🐍 **Click** : Interface CLI pour exécuter les pipelines
- 📈 **MLflow** : Suivi des métriques, modèles et artefacts
- 📦 **DVC** : Versionnement des données et gestion des artefacts
- 🐳 **Docker** : Isolation et reproductibilité des environnements
- 🧪 **Makefile** : Orchestration simplifiée des tâches
- 🌐 **FastAPI** : API pour exposer les prédictions
- ⏳ **Airflow** *(prévu)* : Orchestration des pipelines

---

## 📁 **Arborescence du projet**

```bash
.
├── app/
│   ├── api/
│   │   ├── main.py                # Point d'entrée de l'API FastAPI
│   │   ├── routes/                # Routes de l'API
│   │   ├── services/              # Services (DVC, MLflow, etc.)
│   │   ├── config/                # Configuration de l'application
│   └── mlops/
│       ├── clustering/            # Segmentation des données
│       ├── fusion/                # Jointure des données
│       ├── preprocessing/         # Nettoyage et enrichissement des données
│       ├── Regression/            # Modélisation par régression
│       │   ├── regression_pipeline.py
│       │   ├── train_lgbm.py
│       │   ├── train_xgb.py
│       ├── Serie_temporelle/      # Modélisation par séries temporelles
│       │   ├── seasonal_decomp.py
│       │   ├── sarimax_train.py
│       │   ├── metrics.py
├── data/                          # Données brutes et traitées
├── exports/                       # Artefacts générés (modèles, snapshots)
├── infra/                         # Infrastructure (Docker, CI/CD)
│   ├── deployment/
│   │   ├── docker-compose.yml     # Configuration Docker pour le développement
│   │   ├── docker-compose.prod.yml # Configuration Docker pour la production
├── .github/workflows/             # Pipelines CI/CD GitHub Actions
├── dvc.yaml                       # Pipeline DVC
├── requirements.txt               # Dépendances Python
├── Makefile                       # Commandes simplifiées
├── README.md                      # Ce fichier
```

---

## 📌 **Étapes de la pipeline**

1. **Fusion des données :**
   - Jointure des données DVF et indices socio-économiques.
   - Export des données fusionnées dans `data/`.

2. **Préprocessing :**
   - Nettoyage des données, gestion des valeurs manquantes et encodage.
   - Export des snapshots encodés dans `exports/`.

3. **Clustering :**
   - Segmentation des données avec KMeans.
   - Suivi des métriques et artefacts dans MLflow.

4. **Régression :**
   - Modélisation avec LightGBM et XGBoost.
   - Suivi des performances (`MAE`, `RMSE`, `R²`) dans MLflow.

5. **Séries temporelles :**
   - Modélisation SARIMAX par cluster.
   - Export des modèles dans `exports/st/`.

6. **Suivi des expériences :**
   - Suivi des modèles, métriques et artefacts dans MLflow.
   - Versionnement des données et modèles avec DVC.

---

## 🐳 **Lancement avec Docker**

### **1. Lancer tous les services :**
```bash
docker-compose -f infra/deployment/docker-compose.yml up --build
```

### **2. Lancer en production :**
```bash
docker-compose -f docker-compose.prod.yml up --build
```

### **3. Commandes Makefile :**
- Construire l'image Docker :
  ```bash
  make docker_build
  ```
- Lancer uniquement la régression :
  ```bash
  make docker_run_regression
  ```
- Lancer uniquement les séries temporelles :
  ```bash
  make docker_run_series
  ```

---

## 📊 **Suivi des expériences avec MLflow**

- **Accès à l'interface MLflow :**
  - URL : [http://localhost:5001](http://localhost:5001)

- **Suivi des modèles et métriques :**
  - Modèles : LightGBM, XGBoost, SARIMAX, KMeans.
  - Métriques : `MAE`, `RMSE`, `R²`, `Silhouette score`, etc.
  - Artefacts : Modèles (`.joblib`, `.pkl`), graphiques, données traitées.

---

## 🧪 **Tests et nettoyage**

### **1. Lancer les tests :**
- Tests unitaires et d'intégration :
  ```bash
  pytest app/api/tests/ -v
  ```

### **2. Nettoyage des fichiers générés :**
- Nettoyer les exports :
  ```bash
  make clean_exports
  ```
- Nettoyer le cache DVC :
  ```bash
  make clean_dvc
  ```
- Nettoyer tout :
  ```bash
  make clean_all
  ```

---

## 📦 **Gestion des données avec DVC**

### **1. Initialisation de DVC :**
```bash
dvc init
```

### **2. Ajouter des données :**
```bash
dvc add data/
```

### **3. Synchroniser avec DagsHub :**
- Ajouter un remote :
  ```bash
  dvc remote add -d origin https://dagshub.com/<DAGSHUB_USERNAME>/compagnon-immo.dvc
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user $DAGSHUB_USERNAME
  dvc remote modify origin --local password $DAGSHUB_TOKEN
  ```
- Pousser les données :
  ```bash
  dvc push
  ```

### **4. Récupérer les données :**
```bash
dvc pull
```

---

## 🚀 **Pipeline CI/CD**

- **GitHub Actions :**
  - Tests automatisés (unitaires et d'intégration).
  - Construction et déploiement de l'image Docker.
  - Synchronisation des artefacts avec DagsHub.

- **Commandes principales :**
  - Lancer les tests :
    ```bash
    pytest
    ```
  - Construire et pousser l'image Docker :
    ```bash
    docker build -t ghcr.io/<USERNAME>/compagnon-immo-api:latest .
    docker push ghcr.io/<USERNAME>/compagnon-immo-api:latest
    ```

---

## 📌 **À venir :**
- Intégration complète avec Airflow pour l'orchestration des pipelines.
- Optimisation des performances des modèles.
- Documentation détaillée des endpoints de l'API.

---

## 🛠️ **Contributeurs**
- **Pedro Ketsia** - Développeur principal
- **Collaborateurs** - Merci à tous les contributeurs du projet !

---

## 📄 **Licence**
Ce projet est sous licence MIT. Consultez le fichier `LICENSE` pour plus d'informations.

