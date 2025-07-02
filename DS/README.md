# Projet Compagnon Immobilier : Prévisions des Prix Immobiliers

## 📌 Description du Projet

Ce projet vise à développer une solution permettant aux acheteurs immobiliers d'explorer et comparer des territoires en fonction des prix de l’immobilier et de critères complémentaires tels que la démographie, les transports, les services, l’éducation, la criminalité et l’économie. L’objectif principal est double :

1. **Prédire l’évolution des prix immobiliers par territoire.**
2. **Estimer le prix au m² d'un bien spécifique.**

## 🛠️ Structure du Projet

Le projet est organisé selon la structure suivante :

```
├── LICENSE
├── README.md          <- Le présent fichier.
├── data
│   ├── processed      
│   └── raw            <- Données brutes (annonces, DVF, INSEE, DPE).
│
├── models             <- Modèles entraînés (SARIMAX, LightGBM), sauvegardes et prédictions.
│
├── notebooks          <- Notebooks d’analyse et modélisation (préfixés par ordre).
│                         Exemple : `Part-1 - Exploration - Preprocessing - Split.ipynb`
│
├── references         <- Dictionnaires de données, documentation, sources officielles.
│
├── reports
│   └── figures        <- Graphiques générés (rapport d'exploration des données, visualisations SHAP, diagnostics SARIMAX).
│
├── requirements.txt   <- Dépendances Python du projet (générées avec `pip freeze`).
│
├── src
│   ├── __init__.py
│   ├── features       <- Construction des variables/features à partir des données brutes.
│   │   └── build_features.py
│   ├── models         <- Entraînement et prédiction des modèles.
│   │   ├── train_model.py
│   │   └── predict_model.py
│   ├── visualization  <- Visualisations exploratoires et résultats modèles.
│   │   └── visualize.py
|   ├── streamlit       <- L' application Streamlit de la soutenance.
```
# API d'Estimation Immobilière

## Présentation
Cette API FastAPI permet d'estimer le prix d'un bien immobilier et de fournir une tendance de marché, en s'appuyant sur des modèles Machine Learning réels :
- **LightGBM** pour la prédiction du prix
- **SARIMAX** (un modèle par cluster) pour la tendance de marché

## Organisation des modèles

- Les modèles entraînés doivent être placés dans le dossier : `api_test/models/`
    - Modèle principal LightGBM : `best_lgbm_model.pkl`
    - Modèles SARIMAX par cluster : `best_sarimax_cluster0_parallel.joblib`, `best_sarimax_cluster1_parallel.joblib`, etc.

## Fonctionnement de la sélection SARIMAX

- Lors d'une estimation, le cluster SARIMAX est déterminé dynamiquement à partir de l'adresse (par défaut : cluster 0 pour les codes postaux commençant par 75, sinon cluster 1).
- Le modèle SARIMAX correspondant est utilisé pour la tendance de prix.

## Lancement de l'API

1. **Vérifier la présence des modèles dans `api_test/models/`**
2. **Lancement en local l'API** :
   ```bash
   uvicorn api_test.app.main:app --reload
   ```
3. **Lancement avec Docker**

```bash
docker build -t estimation-api .
docker run -p 8000:8000 --env-file .env estimation-api
```

4. **Modèles Machine Learning**
Placez vos modèles dans le dossier `models/` :
- `models/estimation_lgbm.pkl` (LightGBM)
- `models/evolution_sarimax.joblib` (SARIMAX)

5. **Variables d'environnement**
Créez un fichier `.env` à la racine avec :
```
API_KEY=test-key-123
DATABASE_URL=postgresql://user:password@localhost:5432/estimation
REDIS_URL=redis://localhost:6379/0
```

6. **Documentation interactive**
Accédez à la doc interactive sur :
- http://localhost:8000/docs

7. **Lancement des tests**

- Les tests unitaires sont à adapter pour fonctionner avec les vrais modèles.
- Pour lancer les tests (après adaptation) :
   ```bash
   PYTHONPATH=$(pwd) pytest api_test/app/tests/
   ```

## Personnalisation

- Pour améliorer la logique de sélection du cluster SARIMAX, modifier la fonction `determine_cluster` dans `services/estimation_service.py`.
- Pour ajouter de nouveaux modèles, placer les fichiers dans `api_test/models/` et adapter le code si besoin.