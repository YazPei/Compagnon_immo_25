#!/bin/bash

echo "🏗️Installation des dépendances"
pip install -r requirements.txt

echo "🧪️Etape 1 : Encodage"
python src/encoding.py

echo "🚴‍♀️️ Etape 2 : Entraînement LGBM"
python src/train_lgbm.py

echo "🔍 Etape 3 : Analyse SHAP & résidus"
python src/analyse.py --model lightgbm

