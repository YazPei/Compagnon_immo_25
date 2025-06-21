import streamlit as st

from stutils.utils import load_df_sales_clean_ST, make_evolution_mensuelle_plots, make_evolution_mensuelle_global_plot
from stutils.utils import kmean_par_coude
from stutils.utils import load_train_pour_graph, load_train_pour_graph_cp

#############################################################################################################
# Title
#############################################################################################################

st.title("Analyse exploratoire des données et visualisation")

code = "df.head()"
st.code(code, language="python")

#############################################################################################################
# Load the dataset
#############################################################################################################

df_sales = load_df_sales_clean_ST()

config = {
    "_index": st.column_config.DateColumn("Date", format="MMM YYYY"),
}

st.dataframe(df_sales.sort_index().head(),column_config=config)

#############################################################################################################
# Draw the distribution of the target variable
#############################################################################################################

# Data preparation for the global plot
train_pour_graph = load_train_pour_graph()

# Data preparation for the departments plot
train_pour_graph_cp = load_train_pour_graph_cp()

st.write("## Évolution mensuelle des prix moyens au m² en France")

make_evolution_mensuelle_global_plot(train_pour_graph)

st.write("""
1. **Tendance générale** : On observe une tendance générale à la hausse des prix au mètre carré.
2. **Variations mensuelles** : Les prix montrent des fluctuations mensuelles, avec des pics et des creux à différents moments.
3. **Période de janvier 2020 à janvier 2021** : Durant cette période, les prix ont connu une augmentation progressive avec quelques variations mineures.
4. **Période de janvier 2021 à janvier 2022** : Les prix ont continué à augmenter, mais avec des fluctuations plus marquées.
5. **Période de janvier 2022 à janvier 2023** : Cette période montre une stabilisation relative des prix autour d'un niveau plus élevé, avec des variations moins prononcées que l'année précédente.
6. **Période de janvier 2023 à janvier 2024** : Les prix restent élevés mais montrent une légère tendance à la baisse vers la fin de la période.
7. **Pics et creux** : Les pics les plus élevés sont observés autour de la mi-2022 et de la mi-2023, tandis que les creux les plus bas sont visibles au début de 2020 et à la fin de 2022.
"""
)

st.write('## Évolution mensuelle des prix moyens au m² par département en France')

make_evolution_mensuelle_plots(train_pour_graph_cp)

st.write("### Tendances immobilières par région en France")

# # Normandie
# st.write("#### Normandie")
# st.write("- **Calvados (14)** : Hausse des prix due à la demande touristique.")
# st.write("- **Manche (50)** : Hausse dans les zones côtières.")
# st.write("- **Seine-Maritime (76)** : Hausse autour de Rouen et du Havre.")

# # Bretagne
# st.write("#### Bretagne")
# st.write("- **Morbihan (56)** : Prix les plus élevés, tendance à la hausse.")

# Provence-Alpes-Côte d'Azur
st.write("#### Provence-Alpes-Côte d'Azur")
st.write("- **Bouches-du-Rhône (13)** et **Alpes-Maritimes (06)** : Prix élevés, tendance à la hausse.")
st.write("- **Var (83)** et **Vaucluse (84)** : Prix intermédiaires, tendance à la hausse.")

# # Nouvelle-Aquitaine
# st.write("#### Nouvelle-Aquitaine")
# st.write("- **Gironde (33)** et **Pyrénées-Atlantiques (64)** : Prix élevés, tendance à la hausse.")
# st.write("- **Creuse (23)** : Prix les plus bas, tendance à la hausse modérée.")

# # Bourgogne-Franche-Comté
# st.write("#### Bourgogne-Franche-Comté")
# st.write("- **Côte-d'Or (21)** : Prix élevés, tendance à la hausse.")
# st.write("- **Nièvre (58)** : Prix bas, tendance à la hausse modérée.")

# Île-de-France
st.write("#### Île-de-France")
st.write("- **Paris (75)** : Prix les plus élevés, tendance à la baisse (comportement atypique).")
st.write("- **Hauts-de-Seine (92)** : Prix élevés, tendance à la hausse.")
st.write("- **Seine-Saint-Denis (93)** : Prix bas, tendance à la hausse modérée.")

# # Pays de la Loire
# st.write("#### Pays de la Loire")
# st.write("- **Loire-Atlantique (44)** : Prix les plus élevés, tendance à la hausse.")
# st.write("- **Mayenne (53)** : Prix bas, tendance à la hausse modérée.")

# # Centre-Val de Loire
# st.write("#### Centre-Val de Loire")
# st.write("- **Indre (36)** : Prix bas, tendance à la hausse modérée.")

# # Grand Est
# st.write("#### Grand Est")
# st.write("- **Aube (10)** et **Marne (51)** : Prix élevés, tendance à la hausse.")
# st.write("- **Haute-Marne (52)**, **Meuse (55)**, **Vosges (88)** : Prix bas, tendance à la hausse modérée.")

# # Hauts-de-France
# st.write("#### Hauts-de-France")
# st.write("- **Nord (59)**, **Oise (60)**, **Pas-de-Calais (62)** : Prix élevés, tendance à la hausse.")
# st.write("- **Aisne (02)** : Prix bas, tendance à la hausse.")

# # Occitanie
# st.write("#### Occitanie")
# st.write("- **Hérault (34)** : Prix les plus élevés, tendance à la hausse continue.")
# st.write("- **Haute-Garonne (31)** : Prix élevés, tendance à la hausse.")
# st.write("- **Ariège (09)**, **Lozère (48)** : Prix bas, tendance à la hausse modérée.")

# # Auvergne-Rhône-Alpes
# st.write("#### Auvergne-Rhône-Alpes")
# st.write("- **Savoie (73)** et **Haute-Savoie (74)** : Prix les plus élevés, tendance à la hausse.")
# st.write("- **Loire (42)** et **Haute-Loire (43)** : Prix bas, tendance à la hausse modérée.")

# Comportements atypiques
st.write("#### Comportements atypiques")
st.write("- **Paris (75)** : Tendance à la baisse des prix, contrairement à la plupart des autres départements.")

#############################################################################################################
# Determmine the number of clusters by elbow method
#############################################################################################################

st.write("## Clustering ")

st.write('### Déterminer le nombre de clusters')

st.write("La détermination du nombre optimal de clusters a été effectuée en utilisant la méthode du coude et le score de silhouette")

st.divider()

kmean_by_elbow = kmean_par_coude()

st.write("### 📊 Analyse des Résultats du Clustering")

st.write("#### 🎯 Résultats de l'Analyse")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Méthode du Coude (courbe bleue)**
    - Point d'inflexion à k=5
    - Diminution rapide jusqu'à 5 clusters
    - Stabilisation au-delà de 5
    """)

with col2:
    st.markdown("""
    **Score de Silhouette (courbe rouge)**
    - Score optimal à k=2 (≈0.42)
    - Baisse notable à k=3
    - Stabilisation entre k=4 et k=7
    """)

st.write("#### 💡 Recommandations")

st.markdown("""
**Choix du nombre de clusters:**
- **Segmentation fine** : 5 clusters (selon méthode du coude)
- **Séparation optimale** : 2 clusters (selon score de Silhouette)
- **Compromis** : 4 clusters

**Justification du compromis:**
- Équilibre entre précision et interprétabilité
- Segmentation suffisamment détaillée pour l'analyse
- Maintien d'une bonne cohésion interne des groupes
""")

#############################################################################################################
# dsitribution of the indicators by cluster
#############################################################################################################

st.write("### Distribution des indicateurs par cluster (4) ")

st.image('./images/distrib_cluster.png', width=700)

st.write("### Analyse détaillée des distributions par cluster")

# Caractéristiques par cluster
st.write("#### Caractéristiques principales par cluster")

# Cluster 0
st.write("##### Zone rurale - petite ville stagnante")
st.markdown("""
- Prix les plus bas
- Faible dispersion
- Très faible volatilité
- Croissance faible ou négative
- Typique des zones rurales ou peu dynamiques
""")

# Cluster 1
st.write("##### Centre urbain établi, zone résidentielle")
st.markdown("""
- Prix moyens
- Dispersion moyenne
- Volatilité modérée
- Croissance modérée
- Représentatif des zones périurbaines
""")

# Cluster 2
st.write("##### Banlieue - Zone Mixte")
st.markdown("""
- Prix moyens à moyens-hauts
- Dispersion modérée
- Stabilité relative
- Croissance modérée mais régulière
- Caractéristique des centres urbains établis
""")

# Cluster 3
st.write("##### Zone tendue - ville de luxe")
st.markdown("""
- Prix les plus élevés
- Grande dispersion des valeurs
- Forte volatilité
- Croissance soutenue
- Typique des zones urbaines premium ou touristiques
""")

#############################################################################################################

st.info("""
Cette segmentation permet une compréhension claire des différentes dynamiques du marché immobilier français, 
avec des profils de risque et de rendement bien différenciés.
""")

#############################################################################################################
# Map
#############################################################################################################

st.write('### Visualisation des clusters sur la carte')
st.image('./images/carte_clusters.png', width=800)




