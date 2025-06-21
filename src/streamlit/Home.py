import streamlit as st
from stutils.utils import short_intro_video

# Configuration de la page
st.set_page_config(
    page_title="Compagnon Immo",
    page_icon=":rocket:",
    layout="wide"
)

#############################################################################################################
# Title
#############################################################################################################

st.title("Compagnon Immo")

#############################################################################################################
# Description courte
#############################################################################################################

st.markdown("Ce projet analyse et prédit l'évolution des prix de l'immobilier en France en utilisant des techniques avancées de Machine Learning et d'analyse de séries temporelles.")

#############################################################################################################
# Description
#############################################################################################################

col1, col2 = st.columns(2)

with col1:
    st.subheader("À propos de nous")
    st.write("""
    * Yasmine Peiffer
    * Loick Dernoncourt
    * Christophe Égéa
    * Maxime Hénon
    """)

with col2:
    st.subheader("Les thématiques du projet")
    st.write("""
            - **Preprocessing**
            - **Segmentation** : Clustering K-Means (4 segments identifiés)
            - **Modélisations** : SARIMAX et Prophet
            - **Régression Classique**

    """)

#############################################################################################################
# Petite vidéo
#############################################################################################################

col3, col4, col5 = st.columns(3)

with col4:
    short_intro_video()

#############################################################################################################
# Pied de page
#############################################################################################################