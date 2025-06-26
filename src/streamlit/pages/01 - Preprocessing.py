import streamlit as st
import os
import pandas as pd
from utils.file_upload import load_and_prepare_data, try_read_csv
from utils.data_processing import (
    calculate_missing_percentage,
    plot_missing_values,
    remove_duplicates,
    handle_missing_values,
    identify_categorical_columns,
    plot_categorical_distributions,
    drop_unnecessary_columns,
    process_boolean_columns,
    discretize_annee_construction,
    clean_columns,
    plot_correlation_matrix,
    drop_correlated_columns
)
from utils.outliers_regression import (
    get_numeric_columns,
    plot_boxplots,
    display_outlier_analysis,
    detect_logical_anomalies,
    remove_improbable_values,
    split_train_test,
    process_outliers_sequence,
    plot_distribution
)

# Titre de la page
st.title("Prétraitement des données")
st.write("Veuillez sélectionner votre profil pour continuer.")

# Liste des chemins pour chaque utilisateur
user_paths = {
    "Maxime": "/Users/maximehenon/Documents/GitHub/streamlit-projet-DS/data/raw",
    "Yasmine": "C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON",
    "Loick": "/Users/loick.d/Documents/Datascientest/Github immo/MAR25_BDS_Compagnon_Immo/",
    "Loick MS": "C:/Users/User/Downloads/drive-download-20250508T155351Z-1-001",
    "Christophe": "../data/raw/Sales"
}

# Étape 1 : Sélection du profil
selected_user = st.selectbox("Sélectionnez votre profil :", list(user_paths.keys()))

# Initialiser l'état de validation du profil dans st.session_state
if "profile_validated" not in st.session_state:
    st.session_state["profile_validated"] = False

# Étape 2 : Validation du profil
if not st.session_state["profile_validated"]:
    if st.button("Valider le profil"):
        st.session_state["selected_user"] = selected_user
        st.session_state["profile_validated"] = True
        st.success(f"Profil validé : {selected_user}")
    else:
        st.warning("Veuillez valider votre profil avant de continuer.")
        st.stop()

# Récupérer le chemin correspondant au profil validé
folder_path = user_paths[st.session_state["selected_user"]]

# Construire le chemin complet du fichier
local_file_path = os.path.join(folder_path, "df_sales_clean_polars.csv")


# Fonction de prétraitement des données

@st.cache_data
def preprocess_data(df):
    """
    Effectue toutes les étapes de prétraitement sur les données.
    """
    try:
        # Étape 1 : Chargement des données
        if df is None or df.empty:
            st.error("Le DataFrame est vide ou n'a pas pu être chargé correctement.")
            return None
        
        # Force la colonne INSEE_COM à être de type str
        if "INSEE_COM" in df.columns:
            df["INSEE_COM"] = df["INSEE_COM"].astype(str)
        st.session_state["df_sales_clean"] = df

        # Étape 2 : Suppression des doublons
        if "data_no_duplicates" not in st.session_state:
            st.session_state["data_no_duplicates"] = remove_duplicates(df)

        # Étape 3 : Gestion des valeurs manquantes
        if "data_no_missing" not in st.session_state or "missing_report" not in st.session_state:
            st.session_state["missing_percentage"] = calculate_missing_percentage(st.session_state["data_no_duplicates"])
            st.session_state["missing_values_fig"] = plot_missing_values(st.session_state["missing_percentage"])
            st.session_state["data_no_missing"], st.session_state["missing_report"] = handle_missing_values(st.session_state["data_no_duplicates"])

        # Étape 4 : Identification des colonnes catégoriques
        if "categorical_columns" not in st.session_state or "categorical_distribution_figs" not in st.session_state:
            st.session_state["categorical_columns"] = identify_categorical_columns(st.session_state["data_no_missing"])
            st.session_state["categorical_distribution_figs"] = {
                col: plot_categorical_distributions(st.session_state["data_no_missing"], [col])
                for col in st.session_state["categorical_columns"]
            }

        # Étape 5 : Suppression des colonnes inutiles
        if "data_reduced" not in st.session_state:
            st.session_state["data_reduced"] = drop_unnecessary_columns(st.session_state["data_no_missing"])

        # Étape 6 : Nettoyage des colonnes
        if "data_cleaned" not in st.session_state:
            st.session_state["data_cleaned"] = clean_columns(st.session_state["data_reduced"])

        # Étape 7 : Conversion en booléen
        if "data_boolean" not in st.session_state:
            st.session_state["data_boolean"] = process_boolean_columns(st.session_state["data_cleaned"])

        # Étape 8 : Discrétisation de 'annee_construction'
        if "data_discretized" not in st.session_state:
            st.session_state["data_discretized"] = discretize_annee_construction(st.session_state["data_boolean"])

        # Étape 9 : Matrice de corrélation des variables entièrement corrélées à la cible
        if "correlation_matrix_fig" not in st.session_state:
            st.session_state["correlation_matrix_fig"] = plot_correlation_matrix(
                st.session_state["data_discretized"],
                columns=['prix_bien', 'mensualiteFinance', 'prix_m2_vente'],
                title="Matrice de corrélation"
            )

        # Étape 10 : Suppression des colonnes corrélées
        if "data_no_correlated" not in st.session_state:
            st.session_state["data_no_correlated"] = drop_correlated_columns(st.session_state["data_discretized"], columns_to_drop=['prix_bien', 'mensualiteFinance'])

         # Étape 11 : Calcul des boxplots par défaut
        if "boxplots_fig" not in st.session_state or "numeric_columns" not in st.session_state:
            st.session_state["numeric_columns"] = get_numeric_columns(st.session_state["data_no_correlated"], group_col="INSEE_COM")
            default_columns = ['charges_copro', 'loyer_m2_median_n6', 'loyer_m2_median_n7', 'taux_rendement_n6']
            default_columns = [col for col in default_columns if col in st.session_state["numeric_columns"]]
            st.session_state["boxplots_fig"] = plot_boxplots(st.session_state["data_no_correlated"], default_columns)

        # Étape 12 : Analyse des outliers

        # Étape 13 : Détection des anomalies logiques
        if "data_no_logical_anomalies" not in st.session_state:
            st.session_state["data_no_logical_anomalies"] = detect_logical_anomalies(st.session_state["data_no_correlated"])

        # Étape 14 : Suppression des valeurs improbables
        if "data_final_2" not in st.session_state:
            st.session_state["data_final_2"], st.session_state["improbable_values_report"] = remove_improbable_values(st.session_state["data_no_logical_anomalies"])

        # Étape 15 : Séparation des données en train/test
        if "train_data" not in st.session_state or "test_data" not in st.session_state:
            st.session_state["train_data"], st.session_state["test_data"] = split_train_test(st.session_state["data_final_2"])
            print("Colonnes train après séparation :", st.session_state["train_data"].columns)
            print("Colonnes test après séparation :", st.session_state["test_data"].columns)

        # Vérifiez les shapes après l'étape 15
        print("Shape train_data après séparation :", st.session_state["train_data"].shape)
        print("Shape test_data après séparation :", st.session_state["test_data"].shape)

        # Étape 16 : Gestion des outliers
        if "train_clean" not in st.session_state or "test_clean" not in st.session_state:
            train_clean, test_clean, num_outliers = process_outliers_sequence(
            st.session_state["train_data"],
            st.session_state["test_data"]
        )
        st.session_state["train_clean"] = train_clean
        st.session_state["test_clean"] = test_clean
        st.session_state["num_outliers"] = num_outliers

        # Étape 17 : Visualisation des distributions de la target et de la surface
        if "train_clean" in st.session_state:
            st.session_state["target_distribution_fig"] = plot_distribution(
                st.session_state["train_clean"]["prix_m2_vente"],
                title="Distribution de la target : prix_m2_vente"
            )
            st.session_state["surface_distribution_fig"] = plot_distribution(
                st.session_state["train_clean"]["surface"],
                title="Distribution de la surface"
            )
        else:
            raise ValueError("Les données 'train_clean' ne sont pas disponibles dans st.session_state.")

    except Exception as e:
        st.error(f"Erreur lors du prétraitement des données : {e}")

# Vérifier si un fichier est disponible
uploaded_file = st.file_uploader("Téléchargez un fichier CSV :", type=["csv"])

if uploaded_file or os.path.exists(local_file_path):
    try:
        # Charger les données uniquement si elles ne sont pas déjà chargées
        if "df_sales_clean" not in st.session_state:
            st.session_state["df_sales_clean"] = load_and_prepare_data(uploaded_file, local_file_path)

        # Effectuer le prétraitement uniquement si les résultats ne sont pas déjà calculés
        if "data_no_duplicates" not in st.session_state:  # Vérifie une étape clé pour éviter le recalcul
            preprocess_data(st.session_state["df_sales_clean"])

        # Affichage conditionnel des étapes
        st.sidebar.header("Afficher les étapes")
        show_step_1 = st.sidebar.checkbox("Étape 1 : Chargement des données")
        show_step_2 = st.sidebar.checkbox("Étape 2 : Suppression des doublons")
        show_step_3 = st.sidebar.checkbox("Étape 3 : Gestion des valeurs manquantes")
        show_step_4 = st.sidebar.checkbox("Étape 4 : Identification et distribution des colonnes catégoriques")
        show_step_5 = st.sidebar.checkbox("Étape 5 : Suppression des colonnes inutiles")
        show_step_6 = st.sidebar.checkbox("Étape 6 : Nettoyage et transformation des colonnes dpeL, ges_class et chauffage_energie")
        show_step_7 = st.sidebar.checkbox("Étape 7 : Conversion des colonnes 'porte_digicode', 'ascenseur' et 'cave' en booléen")
        show_step_8 = st.sidebar.checkbox("Étape 8 : Discrétisation de 'annee_construction'")
        show_step_9 = st.sidebar.checkbox("Étape 9 : Matrice de corrélation")
        show_step_10 = st.sidebar.checkbox("Étape 10 : Suppression des colonnes corrélées")
        show_step_11 = st.sidebar.checkbox("Étape 11 : Visualisation des boxplots des variables numériques continues")
        show_step_12 = st.sidebar.checkbox("Étape 12 : Analyse des boxplots")
        show_step_13 = st.sidebar.checkbox("Étape 13 : Détection des anomalies logiques")
        show_step_14 = st.sidebar.checkbox("Étape 14 : Suppression des valeurs improbables")
        show_step_15 = st.sidebar.checkbox("Étape 15 : Séparation des données en train/test")
        show_step_16 = st.sidebar.checkbox("Étape 16 : Gestion des outliers")
        show_step_17 = st.sidebar.checkbox("Étape 17 : Visualisation des distributions de la target et de la surface")

        # Affichage Étape 1
        if show_step_1:
            st.header("Étape 1 : Chargement des données")
            st.write("Shape du DataFrame :", st.session_state["df_sales_clean"].shape)
            st.write("Aperçu des données :")
            st.write(st.session_state["df_sales_clean"].head(5))
            st.write("Origine du fichier :\n " 
            "\n  - Fusion des fichiers de chaque département métropolitain, " 
            "\n - Origine : https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/, " 
            "\n - Taille avant enrichissement : 2,12 Go, " 
            "\n - Taille après enrichissement : 2,84 Go" 
            "\n - Objet de l'enrichissement : \n" 
            "\n       - ajouts de via l'API des DPE (Diagnostic de Performance Energétique), \n" 
            "\n       - ajouts de statistiques INSEE (notamment le niveau des écoles par localité)" )

        if show_step_2:
            st.header("Étape 2 : Suppression des doublons")
            st.write("Shape après suppression des doublons :", st.session_state["data_no_duplicates"].shape)

        # Étape 3 : Gestion des valeurs manquantes
        if show_step_3:
            st.header("Étape 3 : Gestion des valeurs manquantes")
            st.write("Nous éliminons les colonnes ayant plus de 75% de NaNs")
            # Afficher le graphique des valeurs manquantes
            st.pyplot(st.session_state["missing_values_fig"])
            st.write("Shape après gestion des valeurs manquantes :", st.session_state["data_no_missing"].shape)
            st.write("Colonnes éliminées :")
            st.dataframe(st.session_state["missing_report"])

        # Étape 4 : Identification des colonnes catégoriques
        if show_step_4:
            st.header("Étape 4 : Identification des colonnes à moins de 10 modalités (a priori catégoriques)")

            # Vérifiez que les données nécessaires sont disponibles
            if "data_no_missing" in st.session_state and "categorical_columns" in st.session_state:
                # Afficher les colonnes catégoriques identifiées
                categorical_columns = st.session_state["categorical_columns"]
                st.write("Colonnes catégoriques identifiées :", categorical_columns)

                # Colonnes par défaut à afficher
                default_columns = ['type_annonceur', 'typedebien', 'typdebien_lite', 'annonce_exclusive']

                # Vérifier que les colonnes par défaut existent dans les colonnes catégoriques
                default_columns_to_display = [col for col in default_columns if col in categorical_columns]

                # Afficher les graphiques pour les colonnes par défaut
                for col in default_columns_to_display:
                    st.write(f"#### Distribution de la colonne : {col}")
                    # Utiliser les graphiques déjà calculés dans st.session_state
                    for fig in st.session_state["categorical_distribution_figs"][col]:
                        st.pyplot(fig)

                # Permettre à l'utilisateur de sélectionner d'autres colonnes
                st.write("### Sélectionnez des colonnes supplémentaires à afficher")
                additional_columns = st.multiselect(
                    "Colonnes disponibles :",
                    options=[col for col in categorical_columns if col not in default_columns_to_display],
                    default=[]
                )

               # Afficher les graphiques pour les colonnes sélectionnées par l'utilisateur
                if additional_columns:
                    for col in additional_columns:
                        st.write(f"#### Distribution de la colonne : {col}")

                        # Échantillonnage à 20% des données pour accélérer le calcul
                        sampled_data = st.session_state["data_no_missing"].sample(frac=0.2, random_state=42)

                        # Appeler directement la fonction pour générer le graphique sur l'échantillon
                        figures = plot_categorical_distributions(sampled_data, [col])
                        for fig in figures:
                            st.pyplot(fig)
                st.write("Remarques : \n" 
                " \nCertaines variables ont un très grand nombre de modalités, malgré le fait qu’elles soient catégorielles. Nous avons cherché soit à les nettoyer, soit à les discrétiser. \n")
                st.write("À l'opposé, 3 variables n'ont que 2 modalités, et sont donc considérées comme booléennes : porte_digicode, ascenseur et cave. \n" )
            else:
                st.warning("Les données nécessaires pour l'étape 4 ne sont pas disponibles.")

        # Étape 5 : Suppression des colonnes inutiles
        if show_step_5:
            st.header("Étape 5 : Suppression des colonnes inutiles")

            # Vérifiez que les données nécessaires sont disponibles
            if "data_reduced" in st.session_state:
                st.markdown("""
                De manière évidente :
                - **La colonne 'idannonce'** est un identifiant unique pour chaque annonce, elle n'est pas utile pour l'analyse.
                - **La colonne 'annonce_exclusive'** est une variable qui n'est pas utile pour l'analyse.
                - **Les colonnes 'typedebien' et 'typedebien_lite'** contiennent les mêmes informations; nous gardons la plus riche des deux : 'typedebien'.
                - **La colonne 'type_annonceur'** offre une distribution de valeurs trop déséquilibrée.
                - **La colonne 'duree_int'** n'est pas interprétable (valeurs négatives, compréhension empirique).
                - **Les colonnes 'REG', 'DEP', 'IRIS', 'CODE_IRIS', 'TYP_IRIS_x', 'TYP_IRIS_y', 'GRD_QUART', 'UU2010'** sont des colonnes contenant de l'information redondante. De plus, nous créerons une nouvelle colonne pour le code postal, générée à partir des coordonnées géographiques.
                - **La colonne 'INSEE_COM'** est conservée pour l'utiliser comme variable de regroupement lors de la gestion des outliers.
                """)
                st.write("Shape après suppression des colonnes inutiles :", st.session_state["data_reduced"].shape)
            else:
                st.warning("Les données nécessaires pour l'étape 5 ne sont pas disponibles.")

        # Étape 6 : Nettoyage et transformation des colonnes
        if show_step_6:
            st.header("Étape 6 : Nettoyage et transformation des colonnes dpeL, ges_class et chauffage_energie")

            # Vérifiez que les données nécessaires sont disponibles
            if "data_cleaned" in st.session_state:
                st.write("Shape après nettoyage des colonnes :", st.session_state["data_cleaned"].shape)
            else:
                st.warning("Les données nécessaires pour l'étape 6 ne sont pas disponibles.")

        # Étape 7 : Conversion en booléen
        if show_step_7:
            st.header("Étape 7 : Conversion des colonnes 'porte_digicode', 'ascenseur' et 'cave' en booléen")

            # Vérifiez que les données nécessaires sont disponibles
            if "data_boolean" in st.session_state:
                st.write("Shape après conversion en booléen :", st.session_state["data_boolean"].shape)
            else:
                st.warning("Les données nécessaires pour l'étape 7 ne sont pas disponibles.")

        # Étape 8 : Discrétisation de 'annee_construction'
        if show_step_8:
            st.header("Étape 8 : Discrétisation de 'annee_construction'")
            st.write("Nous avons discrétisé la colonne 'annee_construction' en **10** catégories utilisées par data.gouv.")
            st.write("Les périodes correspondent à des changement de normes de construction :")
            st.write("""**_avant 1948    /    1948-1974    /    1975-1977    /    1978-1982    /    1983-1988  
                     1989-2000    /    2001-2005    /    2006-2012    /    2013-2021   /    après 2021_**""")
            
            # Vérifiez que les données nécessaires sont disponibles
            if "data_discretized" in st.session_state:
                st.write("Shape après discrétisation :", st.session_state["data_discretized"].shape)
            else:
                st.warning("Les données nécessaires pour l'étape 8 ne sont pas disponibles.")

        # Étape 9 : Matrice de corrélation
        if show_step_9:
            st.header("Étape 9 : Matrice de corrélation des variables entièrement corrélées à la cible")
            st.write("Cette matrice de corrélation (réduite) permet d'identifier les variables qui sont fortement corrélées entre elles et avec la cible (prix_m2_vente).")

            # Vérifiez que les données nécessaires sont disponibles
            if "data_discretized" in st.session_state:
                # Spécifiez les colonnes à inclure dans la matrice de corrélation
                selected_columns = ['prix_bien', 'mensualiteFinance', 'prix_m2_vente']

                # Vérifiez que les colonnes existent dans le DataFrame
                existing_columns = [col for col in selected_columns if col in st.session_state["data_discretized"].columns]

                if existing_columns:
                    # Générer et afficher la matrice de corrélation
                    fig = plot_correlation_matrix(
                        st.session_state["data_discretized"],
                        columns=existing_columns,
                        title="Matrice de corrélation"
                    )
                    st.pyplot(fig)
                else:
                    # Avertir si les colonnes spécifiées n'existent pas
                    st.warning("Les colonnes spécifiées pour la matrice de corrélation n'existent pas dans le DataFrame.")
            else:
                st.warning("Les données nécessaires pour l'étape 9 ne sont pas disponibles.")

        # Étape 10 : Suppression des colonnes corrélées
        if show_step_10:
            st.header("Étape 10 : Suppression des colonnes prix_bien et mensualiteFinance")

            # Vérifiez que les données nécessaires sont disponibles
            if "data_discretized" in st.session_state:
                st.write("Shape après suppression des colonnes :", st.session_state["data_no_correlated"].shape)
            else:
                st.warning("Les données nécessaires pour l'étape 10 ne sont pas disponibles.")

        # Étape 11 : Visualisation des boxplots avant gestion des outliers
        if show_step_11:
            st.header("Étape 11 : Visualisation des boxplots des variables numériques continues")

            # Vérifiez que les données nécessaires sont disponibles dans st.session_state
            if "boxplots_fig" in st.session_state and "numeric_columns" in st.session_state:
                st.write("Extrait de quelques visualisations :")

                # Récupérer les colonnes par défaut utilisées pour le calcul des boxplots
                default_columns = ['charges_copro', 'loyer_m2_median_n6', 'loyer_m2_median_n7', 'taux_rendement_n6']
                default_columns = [col for col in default_columns if col in st.session_state["numeric_columns"]]

                # Récupérer la figure des boxplots calculée dans preprocess_data
                boxplots_fig = st.session_state["boxplots_fig"]


                # Afficher les boxplots en deux colonnes
                cols = st.columns(2)  # Crée deux colonnes pour l'affichage
                for i, ax in enumerate(boxplots_fig.axes):  # Parcourt les axes de la figure
                    with cols[i % 2]:  # Alterne entre les colonnes
                        st.pyplot(ax.figure)
            else:
                st.warning("Les données nécessaires pour afficher les boxplots ne sont pas disponibles.")

        # Étape 12 : Analyse des outliers
        if show_step_12:
            st.header("Étape 12 : Analyse des boxplots")

            # Vérifiez que les données nécessaires sont disponibles
            display_outlier_analysis()

        # Étape 13 : Détection des anomalies logiques
        if show_step_13:
            st.header("Étape 13 : Détection des anomalies logiques")

            # Vérifiez que les données nécessaires sont disponibles
            if "data_no_logical_anomalies" in st.session_state:
                if st.session_state["data_no_logical_anomalies"].empty:
                    st.info("Aucune anomalie logique détectée.")
                else:
                    st.write("Description des règles de détection :")
                    st.write("""
                        - **Règle 1** : Nombre de toilettes supérieur au nombre de pièces  
                            `(data['nb_toilettes'] > data['nb_pieces'])`

                        - **Règle 2** : Surface trop petite (< 10 m²) ou démesurée (> 1000 m²)  
                            `(data['surface'] < 10) | (data['surface'] > 1000)`

                        - **Règle 3** : Nombre d'étages égal à 0 alors que l'étage est supérieur à 0  
                            `(data['nb_etages'] == 0) & (data['etage'] > 0)`

                        - **Règle 4** : Logement neuf mais année de construction ancienne (avant 2000)  
                            `(data['logement_neuf'] == True) & (data['annee_construction'].isin(["avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988", "1989-2000", "2001-2005", "2006-2012"]))`

                        - **Règle 5** : Prix au mètre carré très bas ou nul  
                            `(data['prix_m2_vente'] < 100)`
                        """)
                    st.write("Shape après suppression des anomalies logiques :", st.session_state["data_no_logical_anomalies"].shape)
            else:
                st.warning("Les données nécessaires pour l'étape 13 ne sont pas disponibles.")

        # Étape 14 : Suppression des valeurs improbables
        if show_step_14:
            st.header("Étape 14 : Suppression des valeurs improbables")

            # Vérifiez que les données nécessaires sont disponibles
            if "data_final_2" in st.session_state:
                st.write("Shape après suppression des valeurs improbables :", st.session_state["data_final_2"].shape)
                st.dataframe(st.session_state["improbable_values_report"])
            else:
                st.warning("Les données nécessaires pour l'étape 14 ne sont pas disponibles.")

        # Étape 15 : Séparation des données en train/test
        if show_step_15:
            st.header("Étape 15 : Séparation des données en train/test")

            if "train_data" in st.session_state and "test_data" in st.session_state:
                # Filtrer les colonnes pour exclure les colonnes *_outlier_flag
                filtered_train_data = st.session_state["train_data"].loc[:, ~st.session_state["train_data"].columns.str.endswith("_outlier_flag")]
                filtered_test_data = st.session_state["test_data"].loc[:, ~st.session_state["test_data"].columns.str.endswith("_outlier_flag")]

                # Afficher les shapes des DataFrames filtrés
                st.write("Shape des données d'entraînement :", filtered_train_data.shape)
                st.write("Shape des données de test :", filtered_test_data.shape)
            else:
                st.warning("Les données nécessaires pour l'étape 15 ne sont pas disponibles.")

        # Étape 16 : Gestion des outliers
        if show_step_16:
            st.header("Étape 16 : Gestion des outliers")
            st.write("""
                        ### Gestion des outliers

                        Une fois les lignes contenant des valeurs improbables éliminées, nous avons géré les outliers par variable en utilisant la méthode de l'écart interquartile (IQR).

                        - **Étape 1** : Identification des outliers  
                        Les outliers sont marqués avec une colonne `OUTLIER_FLAG` par variable concernée, sur les ensembles **train** et **test**.

                        - **Étape 2** : Calcul des médianes  
                        La médiane est calculée par **code INSEE** (équivalent au Code Postal) sur l'ensemble **train** uniquement.

                        - **Étape 3** : Imputation des médianes  
                        Les médianes calculées sont imputées sur les ensembles **train** et **test**.

                        - **Étape 4** : Nettoyage final  
                        Les colonnes `OUTLIER_FLAG` sont supprimées après l'imputation.

                       
                        """)
            # Vérifiez que les données nécessaires sont disponibles
            if "train_clean" in st.session_state and "test_clean" in st.session_state:
                st.write("Shape train après gestion des outliers :", st.session_state["train_clean"].shape)
                st.write("Shape test après gestion des outliers :", st.session_state["test_clean"].shape)

                if "num_outliers" in st.session_state:
                    st.write("Nombre d'outliers détectés (total train et test) :", st.session_state["num_outliers"])
            else:
                st.warning("Les données nécessaires pour l'étape 16 ne sont pas disponibles.")

        # Étape 17 : Visualisation des distributions de la target et de la surface
        if show_step_17:
            st.header("Étape 17 : Visualisation des distributions de la target et de la surface")

            # Vérifiez que les visualisations sont disponibles
            if "target_distribution_fig" in st.session_state and "surface_distribution_fig" in st.session_state:
                st.write("### Distribution de la target : prix_m2_vente")
                st.pyplot(st.session_state["target_distribution_fig"])

                st.write("### Distribution de la surface")
                st.pyplot(st.session_state["surface_distribution_fig"])
            else:
                st.warning("Les visualisations nécessaires pour l'étape 17 ne sont pas disponibles.")

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")
else:
    st.warning("Veuillez uploader un fichier CSV ou vérifier le chemin local par défaut.")
