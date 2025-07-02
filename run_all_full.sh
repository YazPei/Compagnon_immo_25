#!/bin/bash

set -e  # Arrêt immédiat en cas d'erreur

echo "🚀 Lancement du pipeline complet : Régression + Séries temporelles"

# Partie Régression
echo -e "\n🔹 Étape 1 : Lancement du pipeline Régression"
bash mlops/Regression/run_all.sh

# Partie Série Temporelle
echo -e "\n🔸 Étape 2 : Lancement du pipeline Time Series"
bash mlops/Serie_temporelle/run_all_st.sh

echo -e "\n✅ Tous les pipelines ont été exécutés avec succès !"

