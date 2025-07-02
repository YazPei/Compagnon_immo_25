#!/bin/bash


echo "🔁 Lancement du pipeline de régression avec DVC"
dvc repro analyse

echo "📈 Visualisation du graphe"
dvc dag

#echo "🧪️Etape 1 : Encodage"
#python src/encoding.py

#echo "🚴‍♀️️ Etape 2 : Entraînement LGBM"
#python src/train_lgbm.py

#echo "🔍 Etape 3 : Analyse SHAP & résidus"
#python src/analyse.py --model lightgbm

#echo "🔁 Lancement du pipeline de régression avec DVC"
#dvc repro analyse

echo "✅ Pipeline Régression terminée avec succès."

