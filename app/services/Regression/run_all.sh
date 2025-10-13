#!/bin/bash


echo "🔁 Lancement du pipeline de régression avec DVC"
dvc repro analyse

echo "📈 Visualisation du graphe"
dvc dag

#echo "🧪️Etape 1 : Encodage"
#python encoding.py

#echo "🚴‍♀️️ Etape 2 : Entraînement LGBM"
#python train_lgbm.py

#echo "🔍 Etape 3 : Analyse SHAP & résidus"
#python analyse.py --model lightgbm

#echo "🔁 Lancement du pipeline de régression avec DVC"
#dvc repro analyse

echo "✅ Pipeline Régression terminée avec succès."

