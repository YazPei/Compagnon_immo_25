"""File de traitement des données pour le projet Streamlit
   - Chargement des données
   - Suppression des doublons
   - Gestion des valeurs manquantes
   - Identification des colonnes catégoriques
   - Discrétisation de la colonne 'annee_construction'
   - Nettoyage des colonnes
   - Gestion des outliers
   - Normalisation des colonnes
   - Encodage des colonnes catégoriques
   - Sauvegarde du DataFrame nettoyé
   - Affichage des informations sur le DataFrame
   - Affichage des graphiques
   - Affichage des statistiques descriptives
  """



import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

## Paths
folder_path_M = '/Users/maximehenon/Documents/GitHub/streamlit-projet-DS/'

#_----------- ATTENTION LES PATHS NE SONT PAS A JOUR SAUF POUR MAXIME -----------------#
# folder_path_Y = 'C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON'
# folder_path_C = '../data/raw/Sales'
# folder_path_L = '/Users/loick.d/Documents/Datascientest/Github immo/MAR25_BDS_Compagnon_Immo/'
# folder_path_LW = 'C:/Users/User/Downloads/drive-download-20250508T155351Z-1-001'

## Load dataset
input_file = os.path.join(folder_path_M, 'df_sales_clean_polars.csv')
# input_file = os.path.join(folder_path_Y, 'df_sales_clean_polars.csv')
# input_file = os.path.join(folder_path_C, 'df_sales_clean_polars.csv')
# input_file = os.path.join(folder_path_L, 'df_sales_clean_polars.csv')
# input_file = os.path.join(folder_path_LW, 'df_sales_clean_polars.csv')

###########################################################################
############ Lire un fichier CSV avec différents encodages ################
###########################################################################
@st.cache_data

def try_read_csv(path, sep=";", chunksize=100000):
    """
    Tente de lire un fichier CSV avec différents encodages.
    """
    encodings_to_try = ['ISO-8859-1', 'latin1', 'utf-8']
    for encoding in encodings_to_try:
        try:
            print(f"⏳ Tentative d'ouverture avec encodage : {encoding}")
            chunks = pd.read_csv(
                path,
                sep=sep,
                chunksize=chunksize,
                index_col=None,
                on_bad_lines='skip',
                low_memory=False,
                encoding=encoding
            )
            df = pd.concat(chunk for chunk in chunks)
            print(f"✅ Fichier lu avec succès avec encodage : {encoding}")
            return df
        except UnicodeDecodeError as e:
            print(f"⚠️ Échec avec encodage {encoding} : {e}")
        except Exception as e:
            print(f"❌ Autre erreur : {e}")
    raise ValueError("Aucun encodage valide n'a permis d'ouvrir le fichier.")


###########################################################################
##################### ELIMINATION DES DOUBLONS ############################
###########################################################################
def remove_duplicates(data):
    """
    Supprime les doublons dans le dataset.
    """
    return data.drop_duplicates()


###########################################################################
################### GESTION DES VALEURS MANQUANTES ########################
###########################################################################
def calculate_missing_percentage(data):
    """
    Calcule le pourcentage de valeurs manquantes pour chaque colonne.
    """
    missing_percentage = data.isnull().mean() * 100
    missing_value_percentage_sales = pd.DataFrame({
        'column_name': missing_percentage.index,
        'percent_missing': missing_percentage.values
    })
    return missing_value_percentage_sales.sort_values(by='percent_missing', ascending=False)


### GRAPH/VIZ ###
def plot_missing_values(missing_value_percentage_sales):
    """
    Visualise les colonnes avec des valeurs manquantes et retourne une figure Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Tracer le graphique avec une palette personnalisée
    sns.barplot(
        y=missing_value_percentage_sales.column_name,
        x=missing_value_percentage_sales.percent_missing,
        order=missing_value_percentage_sales.column_name,
        ax=ax,
        palette="Blues_r"  # Palette personnalisée
    )
    ax.axvline(x=75, color='red', linestyle='--', label='Threshold (75%)')
    ax.set_title('Répartition des valeurs manquantes dans le dataset', fontsize=12, color='darkblue')
    ax.tick_params(axis='x', labelsize=10, colors='darkgreen')
    ax.tick_params(axis='y', labelsize=10, colors='darkgreen')
    ax.set_ylabel('Features', fontsize=11, color='darkblue')
    ax.set_xlabel('Pourcentage de valeurs manquantes', fontsize=11, color='darkblue')
    ax.legend()
    fig.tight_layout()

    
    return fig

def handle_missing_values(data, threshold=0.75):
    """
    Supprime les colonnes avec un pourcentage de valeurs manquantes supérieur au seuil.
    """
    missing_percentage = data.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    data = data.drop(columns=columns_to_drop)
    return data, columns_to_drop

############################################################################
############### IDENTIFICATION DES COLONNES CATEGORIQUES ###################
############################################################################

def identify_categorical_columns(data, max_unique_values=10):
    """
    Identifie les colonnes catégoriques en fonction du nombre de valeurs uniques.
    """
    categorical_columns = [
        col for col in data.columns if data[col].nunique() <= max_unique_values
    ]
    return categorical_columns


### GRAPH/VIZ ###
def plot_categorical_distributions(data, categorical_columns):
    """
    Trace les distributions des colonnes catégoriques sous forme de graphiques à barres et retourne une liste de figures Matplotlib.
    
    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données.
        categorical_columns (list): Liste des colonnes catégoriques à visualiser.
    
    Returns:
        list: Une liste de figures Matplotlib contenant les graphiques.
    """
    figures = []

    for var_to_viz in categorical_columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=data, x=var_to_viz, ax=ax, palette="Set3")  # Palette personnalisée
        ax.set_title(f'Distribution de {var_to_viz}', fontsize=12, color='darkblue')
        ax.tick_params(axis='x', rotation=45, labelsize=9, colors='darkgreen')
        ax.tick_params(axis='y', labelsize=9, colors='darkgreen')
        ax.set_ylabel('Nombre', fontsize=10, color='darkblue')
        ax.set_xlabel(var_to_viz, fontsize=10, color='darkblue')
        fig.tight_layout()
        figures.append(fig)

    return figures

#############################################################################
########################## SUPPRESSION DES COLONNES INUTILES ################
#############################################################################

def drop_unnecessary_columns(data):
    """
    Supprime les colonnes inutiles du DataFrame.

    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes inutiles supprimées.
    """
    columns_to_drop = [
        'idannonce', 'annonce_exclusive', 'typedebien_lite', 
        'type_annonceur', 'categorie_annonceur',
        'REG', 'DEP', 'IRIS', 'CODE_IRIS', 'TYP_IRIS_x', 'TYP_IRIS_y',
        'nb_logements_copro',
        'GRD_QUART', 'UU2010', 'duree_int'
    ]
    return data.drop(columns=columns_to_drop, axis=1)

# Utilisation de la fonction
# df_sales_clean = drop_unnecessary_columns(df_sales_clean)


############################################################################
########################## CONVERTIT LES COLONNES EN BOOLEEN ###############
############################################################################

def process_boolean_columns(data):
    """
    Convertit les colonnes 'porte_digicode', 'ascenseur' et 'cave' en type booléen.

    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes spécifiées converties en booléen.
    """
    columns = ['porte_digicode', 'ascenseur', 'cave']
    for col in columns:
        if col in data.columns:
            data[col] = data[col].astype(bool)
        else:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")
    
    # Vérification des types des colonnes converties
    print(data[columns].dtypes)
    return data

############################################################################
################# DISCRETISATION ANNEE_CONSTRUCTION ########################
############################################################################


def discretize_annee_construction(data, column_name='annee_construction'):
    """
    Transforme une colonne d'années de construction en une variable catégorielle non ordonnée.
    
    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données.
        column_name (str): Le nom de la colonne à transformer.
    
    Returns:
        pd.DataFrame: Le DataFrame avec la colonne transformée.
    """
    bins = [float('-inf'), 1948, 1974, 1977, 1982, 1988, 2000, 2005, 2012, 2021, float('inf')]
    labels = [
        "avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988",
        "1989-2000", "2001-2005", "2006-2012", "2013-2021", "après 2021"
    ]
    
    if column_name in data.columns:
        data[column_name] = pd.cut(data[column_name], bins=bins, labels=labels, right=False)
    else:
        raise ValueError(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")
    
    return data

############################################################################
############## DISCRETISATION ET NETTOYAGE DPE ET ges_class ################
############################################################################

def clean_classe(val):
    """
    Nettoie et standardise les classes DPE/GES :
    - Vide, "Blank" ou "0" → np.nan
    - Conserve A→G, NS, VI
    - Sinon, extrait un code valide en début de chaîne via regex
    """
    # Handle missing or blank values
    if pd.isna(val) or str(val).strip() in ["", "Blank", "0"]:
        return np.nan
    # Convert to uppercase and remove extra spaces
    s = str(val).strip().upper()
    # Strictly accept known valid codes
    if s in ["A", "B", "C", "D", "E", "F", "G", "NS", "VI"]:
        return s
    # Attempt to extract a valid code at the start of the string
    m = re.match(r"^(NS|VI|[A-G])", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return np.nan

def extract_principal(val):
    """
    Extrait la première source énergétique listée.
    Séparateurs gérés : ',', ';', '/', 'et'.
    """
    if pd.isna(val) or not str(val).strip():  # Cas manquant ou chaîne vide
        return np.nan
    parts = re.split(r"\s*(?:,|;|/|et)\s*", str(val).strip(), maxsplit=1)
    return parts[0] if parts else np.nan


def clean_exposition(val):
    """
    Nettoie et standardise la colonne exposition :
    - Détecte les mots-clés de multi-exposition → "Multi-exposition".
    - Extrait les points cardinaux via regex.
    - Traduit et normalise selon NORM_DIR.
    - Trie et déduplique selon ORDRE_EXPO.
    """
    PATTERN_EXPO = r"(?i)\b(?:Nord(?:-Est|-Ouest)?|Sud(?:-Est|-Ouest)?|Est|Ouest|N|S|E|O)\b"
    ORDRE_EXPO = ["Nord", "Est", "Sud", "Ouest"]
    NORM_DIR = {
        "N": "Nord", "S": "Sud", "E": "Est", "O": "Ouest",
        "NORD": "Nord", "SUD": "Sud", "EST": "Est", "OUEST": "Ouest",
        "NORD-EST": "Nord-Est", "NORD-OUEST": "Nord-Ouest",
        "SUD-EST": "Sud-Est", "SUD-OUEST": "Sud-Ouest"
    }

    if pd.isna(val) or str(val).strip() in ["", "0"]:  # Cas manquant ou valeur vide
        return np.nan
    s = str(val).strip()
    low = s.lower()

    # 1) Multi-exposition via mots-clés
    if any(kw in low for kw in ["traversant", "multi", "toutes", "double expo", "triple", "360"]):
        return "Multi-exposition"

    # 2) Extraction des directions
    matches = re.findall(PATTERN_EXPO, s, flags=re.IGNORECASE)
    dirs = [
        NORM_DIR[m.upper().replace(" ", "-")]
        for m in matches
        if m.upper().replace(" ", "-") in NORM_DIR
    ]

    # 3) Tri et déduplication
    uniq = sorted(set(dirs), key=lambda d: ORDRE_EXPO.index(d.split("-")[0]))
    return "-".join(uniq) if uniq else np.nan


def clean_columns(data):
    """
    Applique les nettoyages spécifiques aux colonnes du DataFrame :
    - Nettoyage des colonnes DPE et GES.
    - Extraction de l'énergie de chauffage principale.
    - Nettoyage de la colonne exposition.
    
    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données.
    
    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    # Nettoyage des colonnes DPE et GES
    for col in ("dpeL", "ges_class"):
        if col in data.columns:
            data[col] = (
                data[col]
                .apply(clean_classe)  # Applique clean_classe à chaque valeur
                .astype("object")     # Force le type chaîne pour les résultats
            )

    # Extraction de l'énergie de chauffage principale
    if "chauffage_energie" in data.columns:
        data["chauffage_energie_principal"] = (
            data["chauffage_energie"]
            .apply(extract_principal)  # Garde la première source énergétique
            .astype("object")
        )
        # Correction de l'encodage mal interprété (ex: Ã -> É)
        data["chauffage_energie_principal"] = (
            data["chauffage_energie_principal"]
            .str.replace("Ã\x89", "É", regex=False)
        )

    # Nettoyage de la colonne exposition
    if "exposition" in data.columns:
        data["exposition"] = (
            data["exposition"]
            .apply(clean_exposition)  # Standardise les orientations
            .astype("object")
        )

    return data

### GRAPH/VIZ ###
def display_cleaned_columns_preview(data):
    """
    Affiche un aperçu des premières lignes des colonnes nettoyées et leurs valeurs uniques.

    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données nettoyées.
    """
    # Affiche un aperçu des premières lignes
    st.write(data[[
        "dpeL",
        "ges_class",
        "chauffage_energie_principal",
        "exposition"
    ]].head(10))

    # Liste des valeurs uniques par colonne
    st.write("Classes DPE :", data["dpeL"].unique())
    st.write("Classes GES :", data["ges_class"].unique())
    st.write("Énergies principales :", data["chauffage_energie_principal"].unique())
    st.write("Expositions :", data["exposition"].unique())

############################################################################
############### SUPPRESSION DES VARIABLES CORRELLEES A LA CIBLE ############
############################################################################


### GRAPH/VIZ ###
def plot_correlation_matrix(data, columns=['prix_bien', 'prix_m2_vente', 'mensualiteFinance'], title='Matrice de corrélation'):
    """
    Trace une matrice de corrélation pour les colonnes spécifiées et retourne une figure Matplotlib.
    """
    correlation_matrix = data[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',  # Palette de couleurs
        fmt=".2f",
        ax=ax,
        cbar_kws={'label': 'Corrélation'}  # Légende pour la barre de couleur
    )
    ax.set_title(title, fontsize=14, color='darkblue')
    fig.tight_layout()
    return fig

def drop_correlated_columns(data, columns_to_drop=['prix_bien', 'mensualiteFinance']):
    """
    Supprime les colonnes corrélées spécifiées du DataFrame.

    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données.
        columns_to_drop (list): Liste des colonnes à supprimer.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes spécifiées supprimées.
    """
    return data.drop(columns=columns_to_drop, axis=1)







################################################################################
########################## ORCHESTRATION DES ETAPES ############################
################################################################################

def process_data(data):
    """
    Orchestration des étapes de transformation des données.
    Ajoute des graphiques pour visualiser les résultats à chaque étape.
    
    Parameters:
        data (pd.DataFrame): Le DataFrame brut.
    
    Returns:
        dict: Un dictionnaire contenant les différentes étapes des données et les graphiques associés.
    """
    results = {}

    # Étape 1 : Suppression des doublons
    data_no_duplicates = remove_duplicates(data)
    results["data_no_duplicates"] = data_no_duplicates
    results["plot_no_duplicates"] = None  # Pas de graphique pour cette étape

    # Étape 2 : Gestion des valeurs manquantes
    data_no_missing, dropped_columns = handle_missing_values(data_no_duplicates)
    results["data_no_missing"] = data_no_missing
    results["plot_missing_values"] = plot_missing_values(calculate_missing_percentage(data_no_duplicates))

    # Étape 3 : Identification des colonnes catégoriques
    categorical_columns = identify_categorical_columns(data_no_missing)
    results["categorical_columns"] = categorical_columns
    results["plot_categorical_distributions"] = plot_categorical_distributions(data_no_missing, categorical_columns)

    # Étape 4 : Nettoyage des colonnes spécifiques
    data_cleaned = clean_columns(data_no_missing)
    results["data_cleaned"] = data_cleaned
    results["plot_cleaned_columns"] = None  # Pas de graphique pour cette étape

    # Étape 5 : Discrétisation de la colonne 'annee_construction'
    data_discretized = discretize_annee_construction(data_cleaned)
    results["data_discretized"] = data_discretized
    results["plot_discretized"] = None  # Pas de graphique pour cette étape

    # Étape 6 : Suppression des colonnes inutiles
    data_reduced = drop_unnecessary_columns(data_discretized)
    results["data_reduced"] = data_reduced
    results["plot_reduced"] = None  # Pas de graphique pour cette étape

    # Étape 7 : Conversion des colonnes en booléen
    data_boolean = process_boolean_columns(data_reduced)
    results["data_boolean"] = data_boolean
    results["plot_boolean"] = None  # Pas de graphique pour cette étape

    # Étape 8 : Suppression des colonnes corrélées
    data_final = drop_correlated_columns(data_boolean)
    results["data_final"] = data_final
    results["plot_correlation_matrix"] = plot_correlation_matrix(data_final)

    return results