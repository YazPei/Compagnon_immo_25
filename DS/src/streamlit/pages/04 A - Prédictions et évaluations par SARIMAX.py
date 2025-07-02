import streamlit as st

from stutils.utils import make_comparison
from stutils.utils import make_sarimax_prediction_cluster_0, make_sarimax_prediction_cluster_1, make_sarimax_prediction_cluster_2, make_sarimax_prediction_cluster_3

#############################################################################################################
# Prédictions pour SARIMAX
#############################################################################################################

st.title("Prédictions des time series avec SARIMAX")

st.write('#### Prédictions pour le cluster "Banlieue - Zone Mixte"')

mae_0, mape_0 = make_sarimax_prediction_cluster_0()

st.write('#### Prédictions pour le cluster "Centre urbain établi, zone résidentielle"')

mae_1, mape_1 = make_sarimax_prediction_cluster_1()

st.write('#### Prédictions pour le cluster "Zone rurale - petite ville stagnante"')
mae_2, mape_2 = make_sarimax_prediction_cluster_2()

st.write('#### Prédictions pour le cluster "Zone tendue - ville de luxe"')
mae_3, mape_3 = make_sarimax_prediction_cluster_3()


#############################################################################################################
# Conclusions
#############################################################################################################

st.write('## Synthèse Globale des Prédictions')

# Tableau comparatif
st.markdown("""
### 📊 Tableau Comparatif des Performances
""")

comparison_data = {
    'Cluster': ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'],
    'MAE (€/m²)': [mae_0, mae_1, mae_2, mae_3],
    'MAE en %': [mape_0, mape_1, mape_2, mape_3],
    'Niveau de Prix': ['Moyen-bas', 'Moyen', 'Moyen-haut', 'Élevé'],
    'Volatilité': ['Faible', 'Moyenne', 'Faible', 'Élevée']
}

make_comparison(comparison_data)

# Analyse générale
st.markdown("""
### 🎯 Points Clés de l'Analyse

#### Forces des Modèles
- **Précision Globale**: MAPE < 5% pour tous les clusters
---

###  Surapprentissage & Généralisation

| Cluster | Zone                           |  MSE Train | MSE Test | MAE Test (€) | Surapprentissage | Dynamique test bien suivie ?             |
|--------:|--------------------------------|----------:|---------:|--------------:|------------------:|-------------------------------------------|
| **0**   |Banlieue parisienne, zone mixte  |  2 163.9   | 1 835.3  | **30.1**      | :x: Non             | :white_check_mark: Oui (légère sous-estimation)            |
| **1**   | Centre urbain établi            |  6 406.7   | 7 167.4  | **72.2**      | :warning: Modéré          | :x: Sous-estimation persistante             |
| **2**   | Zone rurale villes stagnantes   |  1 090.3   | 5 191.9  | **64.5**      | :warning: Modéré          | :x: Prophet : meilleure forme, test mieux   |
| **3**   | Ville spéculative / luxe        |  12 328.9  | 22 610.4 | **130.6**     | :warning: Modéré          | :warning: Partiel (bruit test non anticipé)       |

---
""")