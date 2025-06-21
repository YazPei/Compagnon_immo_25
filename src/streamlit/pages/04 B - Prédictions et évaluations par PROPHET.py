import streamlit as st

from stutils.utils import make_prophet_prediction_cluster_2,make_prophet_prediction_cluster_3


#############################################################################################################
# Prédictions pour PROPHET
#############################################################################################################

st.title("Prédictions des time series avec PROPHET")

st.write('#### Prédictions pour le cluster "Zone rurale - petite ville stagnante"')
make_prophet_prediction_cluster_2()

st.write('#### Prédictions pour le cluster "Zone tendue - ville de luxe"')
make_prophet_prediction_cluster_3()


#############################################################################################################
# Conclusions
#############################################################################################################

# Analyse générale
st.markdown("""
##  📊 Analyse Globale – Modélisation SARIMAX vs Prophet par cluster

###  Analyse détaillée par cluster

####  Cluster 0 – Banlieue parisienne, zone mixte(not. zones frontalières)

- SARIMAX(1,2,1) très stable.
- Erreurs faibles, pas de bruit, résidus propres.
- Pas besoin de Prophet ici.

**Conclusion** :
Modèle parfait pour un marché stable et prévisible.

---

#### Cluster 1 – Centre urbain établi, zone résidentielle

- MSE test > train, sous-estimation persistante.
- SARIMAX ne capture pas les dynamiques locales.

**Suggestions** :
- Ajouter exogènes socio-économiques locales.
- Repenser le clustering (mobilité, revenus, etc.): le cluster est assez dispersé géographiquement
- Travailler avec les équipes métiers pour mieux cerner la spécificité du cluster et le resegmenter (si besoin) en fonction
---

#### Cluster 2 –   Zone rurale 

- **SARIMAX échoue à prédire l’évolution récente.**
- **Prophet capte mieux les ruptures mais reste trop "lisse".**
- Bonne base de tendance, mais pas assez de variabilité.

**Suggestions** :
- **Intégration des changepoints manuels (ex. août 2022).**
- Ajouter saisonnalité

---

####  Cluster 3 – Ville spéculative, luxe

- SARIMAX modélise bien les exogènes mais lisse trop les pics.
- **Prophet suit beaucoup mieux la forme réelle (hausse → baisse).**
- Volatilité captée partiellement avec Prophet + changepoints.

**Suggestions** :
- Utiliser Prophet avec `changepoints` pour ajuster les retournements.
- Éventuellement combiner Prophet (tendance) + modèle ML (résidus).

---

### :scales: Synthèse comparative SARIMAX / Prophet

| Cluster                    | SARIMAX – Forces            | SARIMAX – Limites                | Prophet – Apport principal                          |
|---------------------------:|-----------------------------|----------------------------------|-----------------------------------------------------|
| **Banlieu et zone mixte**  | Très bon suivi, erreurs faibles | Aucun besoin d’amélioration      | :x: Prophet inutile – marché parfaitement modélisé    |
| **Centre urbain établi**   | Base solide, stabilité       | Tendance mal captée              | :soon: À tester                                          |
| **Zone rurale**            | Bon suivi historique         | Mauvaise prévision test          | :white_check_mark: Capte mieux la forme, changepoints efficaces     |
| **Zone tendue**            | Bonne modélisation structurelle | Trop lissé                      | :white_check_mark: Suivi plus précis des retournements et des pics  |

---
""")

# Analyse générale
st.markdown("""

### 💡 Recommandations

1. **Surveillance du Marché**
   - Suivi mensuel renforcé pour les clusters plus volatils
   - Révision trimestrielle des prédictions

---
""")