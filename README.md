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

## 🔬 Méthodologie Résumée

### 1️⃣ Exploration - Preprocessing - Split

* Nettoyage, sélection de variables, gestion des valeurs manquantes, aberrantes et extrêmes.
* Enrichissements via API DPE et INSEE.

### 2️⃣ Encodage et Feature Engineering

* Ordinal, One-hot, Target Encoding.
* Pas de feature selection automatique au final : sélection manuelle par logique métier.

### 3️⃣ Modélisation

* **Séries temporelles (SARIMAX)** : pour prédire l'évolution du prix au m².
* **Régression classique (LightGBM)** : pour estimer le prix au m² à partir de variables.

### 4️⃣ Interprétabilité

* Importance des variables via SHAP.
* Analyse des clusters géographiques pour affiner les performances.

## 🎯 Résultats Clés

* R² de 0.96 et RMSE de 425 €/m² avec LightGBM.
* SARIMAX efficace sur zones stables (rurales/luxe), perfectible sur zones hétérogènes.

## 🚧 Limites et Perspectives

* Incertitude sur la signification des variables : Certaines variables, notamment le rendement, sont fortement corrélées à la cible et ont un comportement cohérent avec leur interprétation économique supposée. Toutefois, l'absence de documentation précise sur leur définition exacte a constitué une limite dans l’analyse causale.
* Extension future possible via NLP et vision par ordinateur sur annonces et images.

## 📈 Utilisation du Projet

* Le projet s’adresse aux potentiels acheteurs immobiliers pour faciliter une prise de décision informée et basée sur des données fiables et compréhensibles.
* Ce projet peut également être un outil d’investissement sur mesure pour accompagner les clients dans une logique de projection à moyen/long terme, en les aidant à prendre position au bon moment sur le marché.

---
## 💾 Installation des dépendances
### Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate  # ou .\\venv\\Scripts\\activate sur Windows

### Installer les dépendances pour reproduire l’environnement du projet :

```bash
pip install -r requirements.txt
```

📝 **Auteur(s)** :

* Yasmine Peiffer
* Loick Dernoncourt
* Christophe Egea
* Maxime Hénon

📅 **Date** : Mars 2025

🔗 **Références** :

* Rapport final complet disponible dans le dossier du projet.

✅ **Licence**
Tous droits réservés. Utilisation autorisée uniquement avec l’accord préalable des auteurs.
