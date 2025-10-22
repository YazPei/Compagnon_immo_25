# 🏠 **Pipeline MLOps - Prédiction des prix immobiliers**

Ce dépôt contient une pipeline complète pour la prédiction des prix immobiliers, divisée en deux approches principales :

- 🔁 **Modélisation par régression** (`LightGBM`, `XGBoost`)
- ⏳ **Modélisation par séries temporelles** (`SARIMAX`)
- 🌐 **API FastAPI** pour servir les modèles et les prédictions

---

## 🧩 **Stack technologique**

- 📈 **MLflow** : Suivi des métriques, modèles et artefacts
- 📦 **DVC** : Versionnement des données et gestion des artefacts
- 🐳 **Docker** : Isolation et reproductibilité des environnements
- 🌐 **FastAPI** : API pour exposer les prédictions
- ⏳ **Airflow** : Orchestration des pipelines

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

### **1. Fusion des données**
- Jointure des données DVF et indices socio-économiques.
- Export des données fusionnées dans `data/`.

### **2. Préprocessing**
- Nettoyage des données, gestion des valeurs manquantes et encodage.
- Export des snapshots encodés dans `exports/`.

### **3. Clustering**
- Segmentation des données avec KMeans.
- Suivi des métriques et artefacts dans MLflow.

### **4. Régression**
- Modélisation avec LightGBM et XGBoost.
- Suivi des performances (`MAE`, `RMSE`, `R²`) dans MLflow.

### **5. Séries temporelles**
- Modélisation SARIMAX par cluster.
- Export des modèles dans `exports/st/`.

### **6. Suivi des expériences**
- Suivi des modèles, métriques et artefacts dans MLflow.
- Versionnement des données et modèles avec DVC.

---

## 🐳 **Lancement avec Docker**

### **1. Lancer tous les services**
```bash
docker-compose -f infra/deployment/docker-compose.yml up --build
```

### **2. Lancer en production**
```bash
docker-compose -f docker-compose.prod.yml up --build
```

### **3. Commandes Makefile**
- Construire l'image Docker :
  ```bash
  make docker-build
  ```
- Lancer uniquement la régression :
  ```bash
  make docker-run-regression
  ```
- Lancer uniquement les séries temporelles :
  ```bash
  make docker-run-series
  ```

---

## 📊 **Suivi des expériences avec MLflow**

- **Accès à l'interface MLflow :**
  - URL : [http://localhost:5050](http://localhost:5050)

- **Suivi des modèles et métriques :**
  - Modèles : LightGBM, XGBoost, SARIMAX.
  - Métriques : `MAE`, `RMSE`, `R²`, etc.
  - Artefacts : Modèles (`.joblib`, `.pkl`), graphiques, données traitées.

---

## 📦 **Gestion des données avec DVC**

### **1. Initialisation de DVC**
```bash
dvc init
```

### **2. Ajouter des données**
```bash
dvc add data/
```

### **3. Synchroniser avec DagsHub**
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

### **4. Récupérer les données**
```bash
dvc pull
```

---

## 🚀 **Pipeline CI/CD**

### **GitHub Actions**
- Tests automatisés (unitaires et d'intégration).
- Synchronisation des artefacts avec DagsHub.

### **Commandes principales**
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

## 📌 **À venir**
- Intégration complète avec Airflow pour l'orchestration des pipelines.
- Optimisation des performances des modèles.
- Documentation détaillée des endpoints de l'API.

---

## Prédiction des Prix Immobiliers par Séries Temporelles

Projet de modélisation des prix immobiliers utilisant les réseaux de neurones récurrents (LSTM/GRU).

## Structure du Projet

```
├── data/                   # Données brutes et traitées
├── models/                 # Modèles de séries temporelles (LSTM/GRU)
├── preprocessing/          # Nettoyage et préparation des données
├── training/              # Scripts d'entraînement
├── evaluation/            # Métriques et visualisations
├── config/                # Fichiers de configuration
├── notebooks/             # Notebooks d'exploration (optionnel)
├── outputs/               # Modèles sauvegardés et résultats
└── main.py               # Point d'entrée principal
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py --data chemin/vers/donnees.csv
```

## Configuration

Modifiez `config/config.yaml` pour ajuster les paramètres du modèle et de l'entraînement.

## Modèles Supportés

- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit

## Métriques d'Évaluation

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient de détermination)
- Précision directionnelle

---

## 🛠️ **Contributeurs**
- **Peiffer Yasmine**
- **Pedro Ketsia**

---

## 📄 **Licence**
Ce projet est sous licence MIT. Consultez le fichier `LICENSE` pour plus d'informations.
```bash
sestatus
```
Pour activer SELinux de façon permanente :
```bash
sudo setenforce 1
sudo sed -i 's/^SELINUX=.*/SELINUX=enforcing/' /etc/selinux/config
```

## Linting des volumes Docker

Pour vérifier que tous les volumes dans vos fichiers `docker-compose.yml` utilisent bien l’option SELinux `:Z` ou `:z`, lancez :

```bash
bash scripts/lint_volumes.sh
```

Intégrez ce script dans votre pipeline CI/CD pour garantir la conformité.

## 🛠️ **Contributeurs**
- **Peiffer Yasmine**
- **Pedro Ketsia**
---

## 📄 **Licence**
Ce projet est sous licence MIT. Consultez le fichier `LICENSE` pour plus d'informations.

