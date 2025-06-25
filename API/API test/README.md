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

1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
2. **Vérifier la présence des modèles dans `api_test/models/`**
3. **Lancer l'API** :
   ```bash
   uvicorn api_test.app.main:app --reload
   ```

## Lancement des tests

- Les tests unitaires sont à adapter pour fonctionner avec les vrais modèles.
- Pour lancer les tests (après adaptation) :
   ```bash
   PYTHONPATH=$(pwd) pytest api_test/app/tests/
   ```

## Personnalisation

- Pour améliorer la logique de sélection du cluster SARIMAX, modifier la fonction `determine_cluster` dans `services/estimation_service.py`.
- Pour ajouter de nouveaux modèles, placer les fichiers dans `api_test/models/` et adapter le code si besoin.

## Contact
Pour toute question : loick.d@datascientest.com 