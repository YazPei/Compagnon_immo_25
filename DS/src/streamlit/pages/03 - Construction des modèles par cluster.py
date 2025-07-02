import streamlit as st

from stutils.utils import make_correlation_matrix, seasonal_decompose_for_clusters, differeciate_cluster
from stutils.utils import plot_acf_pacf, load_train_periodique_q12
from stutils.utils import grid_search_cluster_0, grid_search_cluster_1, grid_search_cluster_2, grid_search_cluster_3

#############################################################################################################
# Title
#############################################################################################################

st.title("Modélisation des séries temporelles")

#############################################################################################################
# Load the dataset
#############################################################################################################

train_periodique_q12 = load_train_periodique_q12()

#############################################################################################################
# Corelation matrix
#############################################################################################################

st.write('### Matrice de corrélation des variables avec la cible')

st.write("L'analyse de la matrice de corrélation des variables par rapport à la variable cible nous a permis de réévaluer nos variables et d'intégrer des notions de décalages temporels (lags)")

make_correlation_matrix(train_periodique_q12)

#############################################################################################################
# Determine if multiplicative or additive series
#############################################################################################################

st.write("## Stationnarité des Séries Temporelles")

st.write("""
### Méthodes utilisées pour Vérifier la Stationnarité
Test de Dickey-Fuller Augmenté (ADF)
### Méthodes pour Atteindre la Stationnarité
1. Différenciation
2. Transformation
3. Décomposition
""")

st.write('### Décomposition des séries temporelles pour les clusters')

seasonal_decompose_for_clusters(train_periodique_q12)

#############################################################################################################
# Stationarization
#############################################################################################################

st.write('### Différenciation et stationnarisation pour les clusters')

clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3 = differeciate_cluster(train_periodique_q12)

#############################################################################################################
# ACF et PACF
#############################################################################################################

st.write('## ACF et PACF')

st.write("Pour déterminer les ordres des modèles, nous avons d'abord procédé manuellement en utilisant " \
"les fonctions d'autocorrélation (ACF) et d'autocorrélation partielle (PACF).")

st.write('### Graphiques ACF et PACF pour les clusters')

plot_acf_pacf(clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3)

#############################################################################################################
# Explication du modèle SARIMAX
#############################################################################################################

st.markdown("""
## Modèle SARIMAX
### Généralité
Le modèle SARIMAX est le modèle le plus général pour la prévision des séries temporelles :
- En l'absence de tendances saisonnières, il devient un modèle ARIMAX.
- Sans variables exogènes, il s'agit d'un modèle SARIMA.
- Sans saisonnalité ni variables exogènes, il devient un modèle ARIMA.
""")

#############################################################################################################
# GridSearch pour SARIMAX
#############################################################################################################

st.write('## GridSearch pour SARIMAX')

st.write("Après plusieurs itérations et en prenant en compte les scores ***AIC*** (Akaike Information Criterion), les résidus et la significativité des ordres, " \
"nous avons eu recours à une recherche par grille (***grid search***) pour les variables exogènes dans le modèle SARIMAX.")

st.write("Le grid search permet d'optimiser les paramètres du modèle. " \
"L'objectif était d'atteindre les meilleurs scores AIC, d'obtenir des résidus " \
"se comportant comme un bruit blanc, et d'assurer une bonne normalité des résidus.")

st.divider()

st.write('## Lancer le GridSearch pour les clusters')

if st.button('Lancer GridSearch'):
    grid_search_cluster_3()
    grid_search_cluster_1()
    grid_search_cluster_2()
    grid_search_cluster_0()


