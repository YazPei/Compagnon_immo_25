import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from utils.Clustering import (
    regroup_cp,
    get_code_postal_final,
    prepare_data_for_clustering,
    calculate_tcam,
    aggregate_monthly_data,
    create_clustering_features,
    perform_clustering,
    apply_clustering_pipeline,
    determine_optimal_clusters,
    plot_cluster_evaluation,
    plot_cluster_distributions,
    plot_cluster_map
)

# Configuration
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Titre de la page
st.title("01.5 - Clustering des Données Immobilières")

# Paramètres généraux
RANDOM_STATE = 42
TARGET_VARIABLE = "prix_m2_vente"

# Chemin du fichier GeoJSON pour les codes postaux
geo_file_path = os.path.join(os.getcwd(), "data/geo/json/contours-codes-postaux.geojson")

# Vérification de l'état de session
if 'train_clean' not in st.session_state or 'test_clean' not in st.session_state:
    st.warning("⚠️ Vous devez d'abord exécuter la page de prétraitement (01-Preprocessing) pour générer les données nettoyées.")
    st.stop()

# Sidebar pour les options
with st.sidebar:
    st.header("Options de Clustering")
    
    # Nombre de clusters
    n_clusters = st.slider(
        "Nombre de clusters :",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
        help="Utilisez le curseur pour explorer l'impact du nombre de clusters sur la qualité de la segmentation"
    )
    
    # Méthode de visualisation
    visualization_method = st.radio(
        "Méthode de visualisation :",
        ["Méthode du coude", "Score de silhouette", "Les deux"],
        index=2
    )
    
    # Bouton pour exécuter le clustering
    run_clustering = st.button("Exécuter le clustering")

# Corps principal
st.header("Introduction au Clustering")

st.write("""
Le clustering est une technique d'apprentissage non supervisé qui permet de regrouper des données similaires en clusters (ou groupes). 
Dans le contexte de l'analyse immobilière, le clustering nous aide à identifier des segments de marché avec des caractéristiques similaires.

Nous utilisons l'algorithme K-means, qui partitionne les données en K clusters, chacun représenté par la moyenne de ses points (centroïde).
""")

# Préparation des données pour le clustering
if run_clustering or 'clustering_done' in st.session_state:
    with st.spinner("Préparation des données pour le clustering..."):
        # Récupération des données d'entraînement et de test
        train_data = st.session_state.train_clean
        test_data = st.session_state.test_clean
        
        # Préparation des données pour le clustering
        try:
            # Étape 1: Regroupement des codes postaux
            st.subheader("1. Regroupement des codes postaux")
            
            # Exemple de regroupement de codes postaux
            sample_codes = ["75001", "13001", "97400", "inconnu"]
            regrouped_codes = [regroup_cp(code) for code in sample_codes]
            
            # Affichage des exemples de regroupement
            regrouped_df = pd.DataFrame({
                "Code postal original": sample_codes,
                "Code postal regroupé": regrouped_codes
            })
            st.write("Exemples de regroupement de codes postaux :")
            st.dataframe(regrouped_df)
            
            st.write("""
            Le regroupement des codes postaux permet de réduire la granularité des données et d'obtenir des segments de marché plus significatifs.
            - Les codes postaux standards sont regroupés par département (2 premiers chiffres)
            - Les codes postaux des DROM-COM (97xxx) sont regroupés par territoire (3 premiers chiffres)
            - Les codes postaux inconnus ou invalides sont regroupés dans la catégorie "inconnu"
            """)
            
            # Étape 2: Préparation des données pour le clustering
            st.subheader("2. Préparation des données pour le clustering")
            
            # Utilisation de la fonction prepare_data_for_clustering
            train_prepared = prepare_data_for_clustering(
                train_data,
                date_col="date",
                code_postal_col="codePostal" if "codePostal" in train_data.columns else "INSEE_COM",
                price_col=TARGET_VARIABLE
            )
            
            test_prepared = prepare_data_for_clustering(
                test_data,
                date_col="date",
                code_postal_col="codePostal" if "codePostal" in test_data.columns else "INSEE_COM",
                price_col=TARGET_VARIABLE
            )
            
            # Affichage des informations sur les données préparées
            st.write(f"Données d'entraînement préparées : {train_prepared.shape[0]} lignes, {train_prepared.shape[1]} colonnes")
            st.write(f"Données de test préparées : {test_prepared.shape[0]} lignes, {test_prepared.shape[1]} colonnes")
            
            # Affichage de la distribution des zones mixtes
            zone_counts = train_prepared["zone_mixte"].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            zone_counts.plot(kind="bar", ax=ax)
            ax.set_title("Top 10 des zones mixtes par nombre de biens")
            ax.set_xlabel("Zone mixte")
            ax.set_ylabel("Nombre de biens")
            st.pyplot(fig)
            
            # Étape 3: Calcul du TCAM par zone
            st.subheader("3. Calcul du Taux de Croissance Annuel Moyen (TCAM) par zone")
            
            # Utilisation de la fonction calculate_tcam
            train_tcam = calculate_tcam(
                train_prepared,
                date_col="date",
                code_postal_col="zone_mixte",
                price_col=TARGET_VARIABLE
            )
            
            # Affichage des TCAM les plus élevés et les plus bas
            st.write("Zones avec les TCAM les plus élevés (croissance forte) :")
            st.dataframe(train_tcam.sort_values("tcam", ascending=False).head(5))
            
            st.write("Zones avec les TCAM les plus bas (croissance faible ou négative) :")
            st.dataframe(train_tcam.sort_values("tcam").head(5))
            
            # Visualisation de la distribution des TCAM
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(train_tcam["tcam"].dropna(), kde=True, ax=ax)
            ax.set_title("Distribution des TCAM par zone")
            ax.set_xlabel("TCAM")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)
            
            # Étape 4: Agrégation des données mensuelles
            st.subheader("4. Agrégation des données mensuelles par zone")
            
            # Utilisation de la fonction aggregate_monthly_data
            monthly_data = aggregate_monthly_data(
                train_prepared,
                date_col="date",
                code_postal_col="zone_mixte",
                price_col=TARGET_VARIABLE
            )
            
            # Affichage des données mensuelles agrégées
            st.write("Aperçu des données mensuelles agrégées :")
            st.dataframe(monthly_data.head())
            
            # Visualisation de l'évolution des prix moyens pour quelques zones
            top_zones = train_tcam.sort_values("tcam", ascending=False).head(3)["zone_mixte"].tolist()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for zone in top_zones:
                zone_data = monthly_data[monthly_data["zone_mixte"] == zone]
                ax.plot(zone_data["date"], zone_data["prix_moyen"], label=f"Zone {zone}")
            
            ax.set_title("Évolution des prix moyens pour les zones à forte croissance")
            ax.set_xlabel("Date")
            ax.set_ylabel("Prix moyen (€/m²)")
            ax.legend()
            st.pyplot(fig)
            
            # Étape 5: Création des features pour le clustering
            st.subheader("5. Création des features pour le clustering")
            
            # Utilisation de la fonction create_clustering_features
            clustering_features = create_clustering_features(
                train_tcam,
                monthly_data,
                price_col="prix_moyen"
            )
            
            # Affichage des features de clustering
            st.write("Aperçu des features de clustering :")
            st.dataframe(clustering_features.head())
            
            # Visualisation des corrélations entre les features
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                clustering_features.drop(columns=["zone_mixte"]).corr(),
                annot=True,
                cmap="coolwarm",
                ax=ax
            )
            ax.set_title("Corrélations entre les features de clustering")
            st.pyplot(fig)
            
            # Étape 6: Détermination du nombre optimal de clusters
            st.subheader("6. Détermination du nombre optimal de clusters")
            
            # Utilisation de la fonction determine_optimal_clusters
            inertia_values, silhouette_values = determine_optimal_clusters(
                clustering_features.drop(columns=["zone_mixte"]),
                max_clusters=10,
                random_state=RANDOM_STATE
            )
            
            # Stockage des résultats dans la session
            st.session_state.inertia_values = inertia_values
            st.session_state.silhouette_values = silhouette_values
            
            # Affichage des résultats
            if visualization_method in ["Méthode du coude", "Les deux"]:
                st.write("### Méthode du coude")
                st.write("""
                La méthode du coude est une technique utilisée pour déterminer le nombre optimal de clusters dans un algorithme de clustering. 
                Elle consiste à tracer l'inertie (somme des distances au carré entre les points et le centre du cluster) en fonction du nombre de clusters 
                et à identifier le point où l'ajout de nouveaux clusters n'améliore plus significativement l'inertie.
                """)
            
            if visualization_method in ["Score de silhouette", "Les deux"]:
                st.write("### Score de silhouette")
                st.write("""
                Le score de silhouette mesure la qualité des clusters en évaluant à quel point chaque point est similaire à son propre cluster (cohésion) 
                par rapport aux autres clusters (séparation). Un score proche de 1 indique que les points sont bien regroupés dans leurs clusters respectifs.
                """)
            
            # Utilisation de la fonction plot_cluster_evaluation
            fig = plot_cluster_evaluation(
                inertia_values,
                silhouette_values,
                max_clusters=10
            )
            st.pyplot(fig)
            
            # Étape 7: Réalisation du clustering
            st.subheader("7. Réalisation du clustering")
            
            # Utilisation de la fonction perform_clustering
            clusters, cluster_centers = perform_clustering(
                clustering_features.drop(columns=["zone_mixte"]),
                n_clusters=n_clusters,
                random_state=RANDOM_STATE
            )
            
            # Ajout des clusters aux features
            clustering_features["cluster"] = clusters
            
            # Affichage de la distribution des clusters
            st.write("Distribution des clusters :")
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
            ax.set_title(f"Distribution des {n_clusters} clusters")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Nombre de zones")
            st.pyplot(fig)
            
            # Étape 8: Application du pipeline de clustering complet
            st.subheader("8. Application du pipeline de clustering complet")
            
            # Utilisation de la fonction apply_clustering_pipeline
            train_clustered, cluster_info = apply_clustering_pipeline(
                train_data, 
                n_clusters=n_clusters,
                date_col="date",
                code_postal_col="codePostal" if "codePostal" in train_data.columns else "INSEE_COM",
                price_col=TARGET_VARIABLE
            )
            
            # Ajout des clusters aux données de test
            zone_cluster_map = dict(zip(cluster_info["zone_mixte"], cluster_info["cluster"]))
            test_clustered = test_prepared.copy()
            test_clustered["cluster"] = test_clustered["zone_mixte"].map(zone_cluster_map)
            
            # Affichage des informations sur les données clusterisées
            st.write(f"Données d'entraînement avec clusters : {train_clustered.shape[0]} lignes, {train_clustered.shape[1]} colonnes")
            st.write(f"Données de test avec clusters : {test_clustered.shape[0]} lignes, {test_clustered.shape[1]} colonnes")
            
            # Stockage des résultats dans la session
            st.session_state.train_clustered = train_clustered
            st.session_state.test_clustered = test_clustered
            st.session_state.cluster_info = cluster_info
            st.session_state.clustering_done = True
            
            st.success("✅ Clustering réalisé avec succès !")
        except Exception as e:
            st.error(f"❌ Erreur lors du clustering : {e}")
            st.stop()

# Affichage des résultats du clustering
if 'clustering_done' in st.session_state:
    st.header("Analyse des résultats du clustering")
    
    # Analyse des résultats
    st.subheader("Interprétation des méthodes d'évaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Méthode du Coude (courbe bleue)**
        - Point d'inflexion visible entre k=3 et k=5
        - Diminution rapide jusqu'à 4-5 clusters
        - Stabilisation au-delà de 5 clusters
        """)
    
    with col2:
        st.markdown("""
        **Score de Silhouette (courbe rouge)**
        - Score optimal généralement entre k=2 et k=4
        - Baisse progressive avec l'augmentation du nombre de clusters
        - Indique une meilleure séparation avec moins de clusters
        """)
    
    st.markdown("""
    **Recommandation:**
    - Pour une segmentation fine : 4-5 clusters
    - Pour une séparation optimale : 2-3 clusters
    - Compromis recommandé : 4 clusters (équilibre entre précision et interprétabilité)
    """)
    
    # Affichage des caractéristiques des clusters
    st.header("Caractéristiques des clusters")
    
    # Distribution des clusters
    st.subheader("Distribution des clusters")
    cluster_counts = st.session_state.train_clustered["cluster"].value_counts().sort_index()
    
    # Création d'un DataFrame pour l'affichage
    cluster_distribution = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Nombre de biens": cluster_counts.values,
        "Pourcentage": (cluster_counts.values / cluster_counts.sum() * 100).round(2)
    })
    
    st.dataframe(cluster_distribution)
    
    # Visualisation de la distribution des clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Cluster", y="Nombre de biens", data=cluster_distribution, ax=ax)
    ax.set_title("Distribution des clusters")
    st.pyplot(fig)
    
    # Distribution des variables par cluster
    st.subheader("Distribution des variables par cluster")
    
    # Utilisation de la fonction plot_cluster_distributions
    cluster_figures = plot_cluster_distributions(
        st.session_state.train_clustered,
        cluster_col="cluster",
        price_col=TARGET_VARIABLE
    )
    
    # Affichage des figures
    for fig in cluster_figures:
        st.pyplot(fig)
    
    # Répartition géographique des clusters
    st.subheader("Répartition géographique des clusters")
    
    # Vérification des colonnes nécessaires
    if all(col in st.session_state.train_clustered.columns for col in ['mapCoordonneesLatitude', 'mapCoordonneesLongitude', 'cluster']):
        # Utilisation de la fonction plot_cluster_map
        map_fig = plot_cluster_map(
            st.session_state.train_clustered,
            lat_col='mapCoordonneesLatitude',
            lon_col='mapCoordonneesLongitude',
            cluster_col='cluster'
        )
        st.pyplot(map_fig)
    else:
        st.warning("Les colonnes nécessaires pour la visualisation géographique ne sont pas disponibles.")
    
    # Interprétation des clusters
    st.header("Interprétation des clusters")
    
    # Calcul des statistiques par cluster
    cluster_stats = st.session_state.train_clustered.groupby("cluster").agg({
        TARGET_VARIABLE: ["mean", "median", "std", "min", "max"],
        "surface": ["mean", "median"] if "surface" in st.session_state.train_clustered.columns else ["count"],
        "nb_pieces": ["mean", "median"] if "nb_pieces" in st.session_state.train_clustered.columns else ["count"]
    }).round(2)
    
    st.dataframe(cluster_stats)
    
    # Description des clusters
    st.subheader("Description des clusters")
    
    # Création d'une description pour chaque cluster
    cluster_descriptions = {
        0: {
            "nom": "Marché économique",
            "description": "Zones à prix bas, faible volatilité, croissance modérée. Typique des zones rurales ou peu dynamiques.",
            "caracteristiques": ["Prix les plus bas", "Faible dispersion", "Très faible volatilité", "Croissance faible ou stable"]
        },
        1: {
            "nom": "Marché intermédiaire",
            "description": "Zones à prix moyens, volatilité modérée, croissance stable. Représentatif des zones périurbaines.",
            "caracteristiques": ["Prix moyens", "Dispersion moyenne", "Volatilité modérée", "Croissance modérée"]
        },
        2: {
            "nom": "Marché dynamique",
            "description": "Zones à prix moyens-hauts, bonne dynamique, croissance soutenue. Caractéristique des centres urbains établis.",
            "caracteristiques": ["Prix moyens à moyens-hauts", "Dispersion modérée", "Stabilité relative", "Croissance régulière"]
        },
        3: {
            "nom": "Marché premium",
            "description": "Zones à prix élevés, forte volatilité, croissance forte. Typique des zones urbaines premium ou touristiques.",
            "caracteristiques": ["Prix les plus élevés", "Grande dispersion", "Forte volatilité", "Croissance soutenue"]
        }
    }
    
    # Affichage des descriptions
    for cluster_id in range(min(n_clusters, 4)):
        if cluster_id in cluster_descriptions:
            st.write(f"**Cluster {cluster_id} - {cluster_descriptions[cluster_id]['nom']}**")
            st.write(cluster_descriptions[cluster_id]['description'])
            st.write("Caractéristiques principales :")
            for carac in cluster_descriptions[cluster_id]['caracteristiques']:
                st.write(f"- {carac}")
            st.write("---")
    
    # Conclusion
    st.header("Conclusion")
    
    st.write("""
    L'analyse par clustering nous a permis d'identifier des segments distincts du marché immobilier français, 
    chacun avec ses propres caractéristiques en termes de prix, volatilité et dynamique de croissance.
    
    Cette segmentation offre une vision claire des différentes dynamiques du marché, avec des profils de risque 
    et de rendement bien différenciés, ce qui peut guider les stratégies d'investissement et les politiques publiques.
    
    Dans la prochaine étape (Régression), nous utiliserons ces clusters comme variables explicatives pour améliorer 
    la précision de nos modèles de prédiction des prix immobiliers.
    """)
    
    # Sauvegarde des données pour la page suivante
    st.session_state.train_data_for_regression = st.session_state.train_clustered
    st.session_state.test_data_for_regression = st.session_state.test_clustered
else:
    st.info("👈 Utilisez les options dans la barre latérale pour configurer et exécuter le clustering.")
