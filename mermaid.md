# Architecture Générale du Pipeline MLOps - Prédiction des Prix Immobiliers

## Vue d'ensemble du Workflow

Ce projet utilise une architecture MLOps complète pour prédire les prix immobiliers avec deux approches :
- **Modélisation par régression** (LightGBM) : prédiction basée sur les caractéristiques du bien
- **Modélisation par séries temporelles** (SARIMAX) : prédiction basée sur l'évolution temporelle des prix

```mermaid
graph TB
    subgraph "Sources de Données"
        A1[DVF - Données de Vente]
        A2[Données Géographiques]
        A3[Indices Socio-économiques]
    end

    subgraph "Infrastructure"
        B1[Docker Compose]
        B2[DVC - Versionnement Données]
        B3[MLflow - Suivi Expériences]
        B4[Airflow - Orchestration]
        B5[PostgreSQL - Métadonnées]
        B6[Redis - Cache]
    end

    subgraph "Pipeline de Données"
        C1[Import Données]
        C2[Fusion Géographique]
        C3[Preprocessing]
        C4[Clustering KMeans]
    end

    subgraph "Branche Régression"
        D1[Encoding des Features]
        D2[Entraînement LightGBM]
        D3[Analyse & Évaluation]
    end

    subgraph "Branche Séries Temporelles"
        E1[Split Train/Test]
        E2[Décomposition Saisonnière]
        E3[Entraînement SARIMAX]
        E4[Évaluation]
    end

    subgraph "Services"
        F1[API FastAPI]
        F2[Monitoring Prometheus]
        F3[Visualisation Grafana]
    end

    A1 --> C1
    A2 --> C2
    A3 --> C2

    B1 --> C1
    B2 --> C1
    B3 --> D2
    B3 --> E3
    B4 --> C1
    B5 --> B4
    B6 --> B4

    C1 --> C2
    C2 --> C3
    C3 --> C4

    C4 --> D1
    D1 --> D2
    D2 --> D3

    C4 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4

    D3 --> F1
    E4 --> F1
    F1 --> F2
    F2 --> F3
```

## Détail du Workflow Étape par Étape

### 1. **Import des Données** (DVC)
- Récupération des données depuis Dagshub/S3
- Gestion des données incrémentales avec checkpoints
- Stockage dans `data/`

### 2. **Fusion Géographique**
- Jointure des données DVF avec les contours postaux
- Enrichissement avec les indices socio-économiques
- Résultat : `data/clean/df_sales_clean.csv`

### 3. **Preprocessing**
- Nettoyage des données (valeurs manquantes, outliers)
- Normalisation et transformation des features
- Séparation train/test
- Sortie : `data/processed/`

### 4. **Clustering**
- Segmentation des données en clusters avec KMeans
- Basé sur les caractéristiques géographiques et économiques
- Préparation pour les modèles spécialisés par cluster

### 5. **Branche Régression**
#### 5.1 Encoding
- Transformation des variables catégorielles
- Normalisation des features numériques
- Sauvegarde des mappings d'encodage

#### 5.2 Entraînement LightGBM
- Modèle de régression par gradient boosting
- Optimisation des hyperparamètres
- Suivi des métriques dans MLflow

#### 5.3 Analyse
- Évaluation sur le jeu de test
- Calcul des métriques (MAE, RMSE, R²)
- Analyse de l'importance des features

### 6. **Branche Séries Temporelles**
#### 6.1 Split
- Division temporelle train/test par cluster
- Préparation des séries chronologiques

#### 6.2 Décomposition
- Analyse saisonnière additive/multiplicative
- Identification des tendances et cycles

#### 6.3 Entraînement SARIMAX
- Modélisation ARIMA avec variables exogènes
- Recherche des meilleurs paramètres par cluster
- Validation croisée temporelle

#### 6.4 Évaluation
- Prédictions sur la période de test
- Calcul des métriques de précision

### 7. **API FastAPI**
- Endpoint `/api/v1/estimation` pour les prédictions
- Support de deux formats de payload (simple/complet)
- Authentification par API key
- Health checks et métriques

### 8. **Monitoring**
- Prometheus pour la collecte de métriques
- Grafana pour les tableaux de bord
- Suivi des performances de l'API

## Architecture des Services Docker

```mermaid
graph LR
    subgraph "Base de Données"
        DB[(PostgreSQL<br/>Airflow)]
        CACHE[(Redis)]
    end

    subgraph "Outils MLOps"
        MLFLOW[MLflow<br/>Port 5050]
        AIRFLOW[Airflow<br/>Webserver 8081<br/>Scheduler]
    end

    subgraph "Pipeline de Traitement"
        DVC[DVC Service]
        FUSION[Fusion Service]
        PREPROC[Preprocessing]
        CLUSTER[Clustering]
        ENCODE[Encoding]
        TRAIN_LGBM[Train LightGBM]
        ANALYSE[Analyse]
        SPLIT[Split ST]
        DECOMP[Decompose ST]
        SARIMAX[Train SARIMAX]
        EVAL[Evaluate ST]
    end

    subgraph "API & Monitoring"
        API[FastAPI<br/>Port 8000]
        PROM[Prometheus<br/>Port 9090]
        GRAF[Grafana<br/>Port 3000]
    end

    subgraph "Tests"
        TEST[API Tests]
        CI[CI Pipeline]
    end

    DB --> AIRFLOW
    CACHE --> AIRFLOW

    AIRFLOW --> DVC
    AIRFLOW --> FUSION
    AIRFLOW --> PREPROC
    AIRFLOW --> CLUSTER
    AIRFLOW --> ENCODE
    AIRFLOW --> TRAIN_LGBM
    AIRFLOW --> ANALYSE
    AIRFLOW --> SPLIT
    AIRFLOW --> DECOMP
    AIRFLOW --> SARIMAX
    AIRFLOW --> EVAL

    MLFLOW --> TRAIN_LGBM
    MLFLOW --> SARIMAX
    MLFLOW --> ANALYSE
    MLFLOW --> EVAL

    TRAIN_LGBM --> API
    SARIMAX --> API

    API --> PROM
    PROM --> GRAF

    API --> TEST
    TEST --> CI
```

## Flux de Données

```mermaid
flowchart TD
    START([Début]) --> IMPORT[Import Données<br/>data/raw/]
    IMPORT --> FUSION[Fusion<br/>data/clean/]
    FUSION --> PREPROC[Preprocessing<br/>data/processed/]
    PREPROC --> CLUSTER[Clustering<br/>exports/df_cluster.csv]

    CLUSTER --> REG_BRANCH{Branche Régression}
    CLUSTER --> ST_BRANCH{Branche Séries Temporelles}

    REG_BRANCH --> ENCODE[Encoding<br/>data/encoded/]
    ENCODE --> TRAIN_LGBM[Train LightGBM<br/>exports/models/]
    TRAIN_LGBM --> ANALYSE[Analyse<br/>metrics/]

    ST_BRANCH --> SPLIT[Split ST<br/>data/split/]
    SPLIT --> DECOMP[Décomposition<br/>exports/st/decomp/]
    DECOMP --> SARIMAX[Train SARIMAX<br/>exports/st/models/]
    SARIMAX --> EVAL[Évaluation<br/>exports/st/eval/]

    ANALYSE --> API[API FastAPI<br/>Port 8000]
    EVAL --> API

    API --> MONITOR[Monitoring<br/>Prometheus/Grafana]
    MONITOR --> END([Fin])

    style START fill:#e1f5fe
    style END fill:#e1f5fe
    style API fill:#c8e6c9
    style MONITOR fill:#fff3e0
```

## Profils Docker Compose

Le projet utilise différents profils pour lancer des sous-ensembles de services :

- **`regression`** : Services pour la modélisation par régression (MLflow, API, tests)
- **`series`** : Services pour les séries temporelles
- **`airflow`** : Orchestration complète avec Airflow
- **`monitoring`** : Prometheus et Grafana
- **`dvc`** : Outils de gestion des données
- **`test`** : Tests automatisés
- **`ci`** : Pipeline d'intégration continue

## Points d'Accès

- **API FastAPI** : http://localhost:8000
  - Documentation : http://localhost:8000/docs
  - Health check : http://localhost:8000/api/v1/health

- **MLflow** : http://localhost:5050
  - Interface de suivi des expériences

- **Airflow** : http://localhost:8081
  - Interface d'orchestration des pipelines

- **Grafana** : http://localhost:3000
  - Tableaux de bord de monitoring (admin/admin)

- **Prometheus** : http://localhost:9090
  - Métriques système

Cette architecture permet une séparation claire des responsabilités, une reproductibilité grâce à Docker, et un suivi complet des expériences avec MLflow et DVC.
