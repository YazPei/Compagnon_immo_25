# Projet Streamlit - Analyse et Prétraitement des Données

Ce projet Streamlit vise à analyser, prétraiter, visualiser des données immobilières puis entrainer des modèles permettant de faire des prévisions sur le prix de vente au m2 en France, à partir de données agrégées des 5 dernieres années de ventes immobilières sur le territoire métropolitain.

Deux types de modélisations font suite aux analyses et prétraitements : 
 - une modélisation de régression
 - une modélisation via une série temporelle

Il est structuré pour séparer les différentes étapes du traitement des données, de leur chargement à leur visualisation, en passant par le nettoyage et la détection des anomalies.

---

## Fonctionnalités

### 1. Prétraitement des Données
- Validation du profil utilisateur pour sélectionner les chemins de données.
- Chargement des données brutes depuis un fichier local ou un fichier téléchargé.
- Suppression des doublons et gestion des valeurs manquantes.
- Identification et transformation des colonnes catégoriques et numériques.
- Gestion des outliers avec la méthode de l'écart interquartile (IQR).
- Suppression des valeurs improbables et anomalies logiques.

### 2. Visualisation
- Matrice de corrélation pour analyser les relations entre les variables.
- Visualisation des distributions des variables numériques (boxplots, histogrammes).
- Analyse des distributions de la target et des variables explicatives.

### 3. Modélisation et Évaluation
- Implémentation et évaluation d'un modèle **LightGBM**.
- Calcul des métriques de performance : **R²**, **MAE**, **RMSE**.
- Analyse des contributions des variables avec **SHAP** (SHapley Additive exPlanations).

---

## Structure du Projet

``` bash
streamlit-projet-DS/
├── data/ # Folder containing raw and cleaned data
├── images/
├── models/ # Folder containing SARIMAX models
├── output/
├── pages/ # Streamlit files for each step of the project
│   ├── 01_Preprocessing.py # Data preprocessing
│   ├── 02_Exploration_des_TimeSeries.py # Time series exploration
│   ├── 03_Construction_des_modeles_par_cluster.py # Model building by cluster
│   ├── 04_A_Predictions_et_evaluations_par_SARIMAX.py # Predictions and evaluations by SARIMAX
│   ├── 04_B_Predictions_et_evaluations_par_PROPHET.py # Predictions and evaluations by PROPHET
│   └── 05_Regression_LGBM_et_evaluation_du_modele.py # LGBM regression and model evaluation
├── sutils/
│   └── utils.py # Utility functions
├── utils/ # Utility functions
│   ├── outliers_regression.py # Outlier management
│   ├── file_upload.py # File loading
│   └── visualizations.py # Visualization functions
├── videos/
└── requirements.txt # Python dependencies
```

## Installation

### Prérequis
- Python 3.8 ou supérieur
- **pip** ou **conda** pour gérer les dépendances

### Étapes
1. Clonez le repository :
   ```bash
   git clone https://github.com/<votre-utilisateur>/streamlit-projet-DS.git
   cd streamlit-projet-DS

2. Installez les dépendances :

pip install -r requirements.txt

3. Lancez l'application Streamlit :

streamlit run pages/01\ -\ Preprocessing.py ou streamlit run Home.py

### Utilisation
Étape 1 : Prétraitement des données

Sélectionnez votre profil utilisateur.
Chargez les données brutes ou utilisez le fichier local par défaut.
Suivez les étapes de nettoyage et de transformation.


Étape 2 : Visualisation

Explorez les relations entre les variables à l'aide de graphiques interactifs.


Étape 3 : Modélisation

Construisez et entraînez un modèle LightGBM.
Évaluez les performances du modèle.


Étape 4 : Analyse SHAP

Analysez les contributions des variables explicatives au modèle.


Étape 5 : Validation

Effectuez une validation croisée pour évaluer la robustesse du modèle.


Étape 6 : Évaluation finale

Obtenez les métriques finales et visualisez les performances.


Auteurs
Yasmine Peiffer
Loick Dernoncourt
Christophe Égéa
Maxime Henon Hénon

Licence
Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus d'informations.

Remarques
Si vous rencontrez des problèmes ou avez des suggestions, n'hésitez pas à ouvrir une issue sur le repository GitHub.
