import pandas as pd
import numpy as np
import streamlit as st
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import streamlit as st
import seaborn as sns


################################################################################
########################## FILTRAGE DES COLONNES NUMERIQUES ####################
################################################################################
def get_numeric_columns(data, group_col, excluded_cols=None):
    """
    Retourne les colonnes numériques en excluant une colonne de regroupement et d'autres colonnes spécifiées.

    Parameters:
        data (pd.DataFrame): Le DataFrame à analyser.
        group_col (str): La colonne utilisée pour regrouper les données (sera exclue).
        excluded_cols (list): Liste des colonnes supplémentaires à exclure (par défaut, coordonnées géographiques).

    Returns:
        list: Liste des colonnes numériques restantes après exclusion.
    """
    if excluded_cols is None:
        excluded_cols = ['mapCoordonneesLatitude', 'mapCoordonneesLongitude']

    # Vérifier si group_col existe dans le DataFrame
    if group_col not in data.columns:
        raise ValueError(f"La colonne de regroupement '{group_col}' n'existe pas dans le DataFrame.")

    # Vérifier si les colonnes à exclure existent dans le DataFrame
    excluded_cols = [col for col in excluded_cols if col in data.columns]

    # Sélectionner les colonnes numériques en excluant group_col et excluded_cols
    numeric_columns = [
        col for col in data.select_dtypes(include='number').columns
        if col != group_col and col not in excluded_cols
    ]

    # Vérification finale : s'assurer qu'il reste des colonnes numériques
    if not numeric_columns:
        raise ValueError("Aucune colonne numérique valide n'a été trouvée après exclusion.")

    return numeric_columns

####################################################################################
########################## AFFICHAGE DES BOXPLOTS ##################################
####################################################################################

def plot_boxplots(data, selected_columns):
    """
    Trace des boxplots pour les colonnes sélectionnées et retourne un objet figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Créer une figure pour les boxplots
    fig, axes = plt.subplots(nrows=len(selected_columns), ncols=1, figsize=(8, len(selected_columns) * 4))

    # Si une seule colonne est sélectionnée, axes n'est pas une liste
    if len(selected_columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, selected_columns):
        sns.boxplot(data=data, y=col, ax=ax, palette="Set2", width=0.5, linewidth=1.5)
        ax.set_title(f"Boxplot : {col}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Valeurs", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

##################################################################################
########################## ANALYSE DES RÉSULTATS #################################
##################################################################################

def display_outlier_analysis():
    """
    Affiche une analyse des étapes de traitement des outliers avec une mise en page esthétique.
    """
    st.markdown("""
    Les **boxplots** révèlent des distributions surprenantes et mettent en évidence des problèmes potentiels d'unités d'échelle pour certaines variables.
    Voici une liste non exhaustive des variables concernées :
    """)

    # Liste des variables problématiques
    st.markdown("""
    - `charges_copro`
    - `loyer_m2_median_n6`
    - `loyer_m2_median_n7`
    - `taux_rendement_n6`
    - `taux_rendement_n7`
    - `nb_log_n6`
    - `nb_log_n7`
    """)

    st.markdown("""
    ### Étapes prévues pour traiter les outliers :
    1. **Détection d'anomalies logiques** : Éliminer les valeurs incohérentes selon des règles prédéfinies.
    2. **Détection d'anomalies visuelles** : Identifier les valeurs aberrantes à l'aide des boxplots.
    3. **Séparation des données** : Diviser les données en ensembles d'entraînement et de test pour éviter le data leakage.
    4. **Traitement des valeurs extrêmes** :
        - Détection des outliers.
        - Imputation des valeurs aberrantes par la médiane (par code INSEE).
    """)

    

#################################################################################
########################## DÉTECTION DES ANOMALIES LOGIQUES ########################
#################################################################################


def detect_logical_anomalies(data):
    """
    Supprime les lignes contenant des anomalies logiques dans un DataFrame selon des règles prédéfinies.

    Args:
        data (pd.DataFrame): Le DataFrame à analyser.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé des lignes contenant des anomalies logiques.
    """
    # Liste des règles d'anomalies logiques
    rules = [
        # Règle 1 : nb_toilettes > nb_pieces (pas logique dans un logement classique)
        (data['nb_toilettes'] > data['nb_pieces']),
        
        # Règle 2 : surface trop petite (< 10 m²) ou démesurée (> 1000 m²)
        (data['surface'] < 10) | (data['surface'] > 1000),
        
        # Règle 3 : nb_etages = 0 alors que etage > 0 (impossible sans étage)
        (data['nb_etages'] == 0) & (data['etage'] > 0),
        
        # Règle 4 : logement neuf mais année de construction ancienne (avant 2000)
        (data['logement_neuf'] == True) & (
            data['annee_construction'].isin([
                "avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988", "1989-2000"])),
        
        # Règle 5 : prix_m2_vente très bas ou nul 
        (data['prix_m2_vente'] < 100)
    ]

    # Combiner toutes les règles pour identifier les lignes à supprimer
    combined_rule = pd.concat(rules, axis=1).any(axis=1)

    # Supprimer les lignes contenant des anomalies logiques
    data = data[~combined_rule].copy()

    # Résumé : Nombre total de lignes supprimées
    nb_anomalies = combined_rule.sum()
    print(f"{nb_anomalies} lignes contenant des anomalies logiques ont été supprimées.")

    return data

####################################################################################
########################## DÉTECTION DES ANOMALIES DE SAISIE #######################
####################################################################################

def remove_improbable_values(data):
    """
    Supprime les lignes contenant des valeurs improbables selon des seuils définis
    et retourne un DataFrame nettoyé ainsi qu'un rapport des suppressions.

    Args:
        data (pd.DataFrame): Le DataFrame à analyser.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
        pd.DataFrame: Un rapport des suppressions par colonne.
    """
    # Définir les colonnes suspectes et les seuils
    cols_suspectes = ['charges_copro', 'loyer_m2_median_n6', 'loyer_m2_median_n7', 
                      'taux_rendement_n6', 'taux_rendement_n7', 'nb_log_n6', 'nb_log_n7']
    seuils_max = {
        'charges_copro': 10000,
        'loyer_m2_median_n6': 100,
        'loyer_m2_median_n7': 100,
        'taux_rendement_n6': 50,
        'taux_rendement_n7': 50,
        'nb_log_n6': 1000,
        'nb_log_n7': 1000
    }
    seuils_min = {
        'charges_copro': 0,
        'loyer_m2_median_n6': 5,
        'loyer_m2_median_n7': 5,
        'taux_rendement_n6': 0,
        'taux_rendement_n7': 0,
        'nb_log_n6': 1,
        'nb_log_n7': 1
    }

    problemes = {}
    mask_valeurs_improbables = pd.Series(False, index=data.index)  # Initialiser le masque

    for col in cols_suspectes:
        # Détection des valeurs au-dessus du seuil maximum
        if col in seuils_max:
            mask_above = data[col] > seuils_max[col]
            mask_valeurs_improbables |= mask_above
            n_anormaux_max = mask_above.sum()
        else:
            n_anormaux_max = 0

        # Détection des valeurs en dessous du seuil minimum
        if col in seuils_min:
            mask_below = data[col] < seuils_min[col]
            mask_valeurs_improbables |= mask_below
            n_anormaux_min = mask_below.sum()
        else:
            n_anormaux_min = 0

        # Ajouter au rapport si des valeurs aberrantes sont détectées
        if n_anormaux_max > 0 or n_anormaux_min > 0:
            problemes[col] = {
                'nb_anormaux_max': n_anormaux_max,
                'max_valeur': data[col].max(),
                'nb_anormaux_min': n_anormaux_min,
                'min_valeur': data[col].min()
            }

    # Création d'un DataFrame pour le rapport
    df_problemes = pd.DataFrame.from_dict(problemes, orient='index')

    # Supprimer les lignes contenant des valeurs improbables
    data = data[~mask_valeurs_improbables].copy()

    # Nombre total de lignes supprimées
    nb_lignes_supprimees = mask_valeurs_improbables.sum()
    print(f"{nb_lignes_supprimees} lignes contenant des valeurs improbables ont été supprimées.")
     # Vérifiez les colonnes après suppression des valeurs improbables
    print("Colonnes après suppression des valeurs improbables :", data.columns)

    return data, df_problemes



###################################################################################
########################## SPLIT TRAIN/TEST ######################################
####################################################################################    

def split_train_test(data):
    """
    Splits the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset to split.

    Returns:
        tuple: A tuple containing the training set (train_data) and the testing set (test_data).
    """
   
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data
    



###################################################################################
########################## TRAITEMENT DES OUTLIERS ################################
###################################################################################


def process_outliers_sequence(
    train_data,
    test_data,
    group_col="INSEE_COM",
    lower_perc=0.01,
    upper_perc=0.99,
    outlier_tag=-999,
    target_col="prix_m2_vente"
):
    """
    Traite les outliers dans une séquence complète en prenant en entrée train_data et test_data.
    
    Parameters:
        train_data (pd.DataFrame): Données d'entraînement.
        test_data (pd.DataFrame): Données de test.
        group_col (str): Colonne utilisée pour regrouper les données (par défaut "INSEE_COM").
        lower_perc (float): Quantile inférieur pour détecter les outliers (par défaut 0.01).
        upper_perc (float): Quantile supérieur pour détecter les outliers (par défaut 0.99).
        outlier_tag (int): Valeur utilisée pour marquer les outliers (par défaut -999).
        target_col (str): Nom de la colonne cible à séparer (par défaut "prix_m2_vente").
    
    Returns:
        tuple: train_clean, test_clean (données nettoyées pour train et test).
    """
    # Étape 1 : Obtenir les colonnes numériques en excluant les coordonnées géographiques
    excluded_cols = ['mapCoordonneesLatitude', 'mapCoordonneesLongitude']
    numeric_cols = [
        col for col in train_data.select_dtypes(include='number').columns
        if col != group_col and col not in excluded_cols
    ]
    print(f"Colonnes numériques sélectionnées : {numeric_cols}")

    # Étape 2 : Calculer les bornes à partir des données d'entraînement
    bounds = {
        col: (
            train_data[col].quantile(lower_perc),
            train_data[col].quantile(upper_perc)
        )
        for col in numeric_cols
    }
    print(f"Bornes calculées pour les colonnes numériques : {bounds}")

    # Étape 3 : Calculer les médianes de groupe et globales à partir de train_data
    group_medians = {
        col: train_data.groupby(group_col)[col].median()
        for col in bounds
    }
    global_medians = train_data[list(bounds)].median()
    print(f"Médianes par groupe : {group_medians}")
    print(f"Médianes globales : {global_medians}")

    # Initialiser les compteurs d'outliers
    num_outliers_train = 0
    num_outliers_test = 0

    # Étape 4 : Marquer les outliers dans train_data
    for col, (low, high) in bounds.items():
        mask = (train_data[col] < low) | (train_data[col] > high)
        num_outliers_train += mask.sum()
        train_data[f'{col}_outlier_flag'] = mask.astype(int)
        train_data.loc[mask, col] = outlier_tag

    # Étape 5 : Supprimer les lignes où la target est un outlier
    mask_train_keep = train_data[f'{target_col}_outlier_flag'] == 0
    train_data = train_data[mask_train_keep]

    # Étape 6 : Nettoyer les outliers dans train_data
    for col in bounds:
        mask = train_data[col] == outlier_tag
        if mask.any():
            train_data.loc[mask, col] = (
                train_data.loc[mask, group_col]
                    .map(group_medians[col])
                    .fillna(global_medians[col])
                    .astype(train_data[col].dtype)
            )
    print(f"Exemple de données nettoyées (train_clean) :\n{train_data.head()}")

    # Étape 7 : Marquer les outliers dans test_data
    for col, (low, high) in bounds.items():
        mask = (test_data[col] < low) | (test_data[col] > high)
        num_outliers_test += mask.sum()
        test_data[f'{col}_outlier_flag'] = mask.astype(int)
        test_data.loc[mask, col] = outlier_tag
    print(f"Exemple de données marquées comme outliers (test_data) :\n{test_data.head()}")

    # Étape 8 : Supprimer les lignes où la target est un outlier dans test_data
    mask_test_keep = test_data[f'{target_col}_outlier_flag'] == 0
    test_data = test_data[mask_test_keep]

    # Étape 9 : Nettoyer les outliers dans test_data
    for col in bounds:
        mask = test_data[col] == outlier_tag
        if mask.any():
            test_data.loc[mask, col] = (
                test_data.loc[mask, group_col]
                    .map(group_medians[col])
                    .fillna(global_medians[col])
                    .astype(test_data[col].dtype)
            )

    # Suppression des colonnes de marquage
    train_data.drop(columns=[f'{col}_outlier_flag' for col in bounds], inplace=True)
    test_data.drop(columns=[f'{col}_outlier_flag' for col in bounds], inplace=True)

    # Vérification finale : s'assurer qu'il ne reste aucune valeur outlier_tag
    assert not train_data.isin([outlier_tag]).any().any(), "Des valeurs outlier_tag sont encore présentes dans train_clean."
    assert not test_data.isin([outlier_tag]).any().any(), "Des valeurs outlier_tag sont encore présentes dans test_clean."


    # Calcul du nombre total d'outliers gérés
    total_outliers = num_outliers_train + num_outliers_test

    # Afficher un résumé des données nettoyées
    print(f"train_clean shape: {train_data.shape}")
    print(f"test_clean shape: {test_data.shape}")
    print(f"Exemple train_clean:\n{train_data.head()}")
    print(f"Exemple test_clean:\n{test_data.head()}")

    return train_data, test_data, total_outliers


######################################################################################
########################## Visualisations des distributions ################################
def plot_distribution(column_data, title):
    """
    Génère un histogramme avec une courbe KDE pour visualiser la distribution d'une colonne.

    Parameters:
        column_data (pd.Series): Données de la colonne à visualiser.
        title (str): Titre du graphique.

    Returns:
        matplotlib.figure.Figure: La figure contenant l'histogramme et la courbe KDE.
    """
    if column_data is None or len(column_data) == 0:
        raise ValueError("La colonne est vide ou non définie.")

    # Supprimer les valeurs NaN ou infinies
    column_data = column_data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(column_data) == 0:
        raise ValueError("La colonne ne contient que des valeurs valides après nettoyage.")

    # Générer la figure
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=column_data, bins=150, kde=True, ax=ax, color="blue")
    ax.set_title(title, fontsize=14)

    # Ajuster les limites de l'axe X en fonction des valeurs minimales et maximales
    ax.set_xlim(column_data.min(), column_data.max())

    # Ajuster les ticks
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    return fig

################################################################################
########################## ORCHESTRATION DES ETAPES ############################
################################################################################
def process_outliers(data, group_col="INSEE_COM", target_col="prix_m2_vente", lower_perc=0.01, upper_perc=0.99, outlier_tag=-999):
    """
    Orchestration des étapes de traitement des outliers.
    Ajoute des graphiques pour visualiser les résultats à chaque étape.
    
    Parameters:
        data (pd.DataFrame): Le DataFrame brut.
        group_col (str): Colonne utilisée pour regrouper les données (par défaut "INSEE_COM").
        target_col (str): Colonne cible à séparer (par défaut "prix_m2_vente").
        lower_perc (float): Quantile inférieur pour détecter les outliers (par défaut 0.01).
        upper_perc (float): Quantile supérieur pour détecter les outliers (par défaut 0.99).
        outlier_tag (int): Valeur utilisée pour marquer les outliers (par défaut -999).
    
    Returns:
        dict: Un dictionnaire contenant les différentes étapes des données et les graphiques associés.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    results = {}

    # Étape 1 : Visualisation initiale des boxplots
    numeric_cols = get_numeric_columns(data, group_col)
    results["numeric_columns"] = numeric_cols
    fig_initial_boxplots = plot_boxplots(data, numeric_cols)
    results["plot_initial_boxplots"] = fig_initial_boxplots

    # Étape 2 : Suppression des anomalies logiques
    data_cleaned_logical = detect_logical_anomalies(data)
    results["data_cleaned_logical"] = data_cleaned_logical
    results["plot_logical_anomalies"] = None  # Pas de graphique pour cette étape

    # Étape 3 : Suppression des valeurs improbables
    data_cleaned_improbable, report_improbable = remove_improbable_values(data_cleaned_logical)
    results["data_cleaned_improbable"] = data_cleaned_improbable
    results["report_improbable"] = report_improbable
    results["plot_improbable_values"] = None  # Pas de graphique pour cette étape

    # Étape 4 : Split train/test
    train_data, test_data = split_train_test(data_cleaned_improbable)
    results["train_data"] = train_data
    results["test_data"] = test_data

    # Étape 5 : Traitement des outliers
    train_clean, test_clean = process_outliers_sequence(
        train_data=train_data,
        test_data=test_data,
        group_col=group_col,
        lower_perc=lower_perc,
        upper_perc=upper_perc,
        outlier_tag=outlier_tag,
        target_col=target_col
    )
    results["train_clean"] = train_clean
    results["test_clean"] = test_clean

    # Étape 6 : Visualisation des boxplots pour train_clean
    fig_boxplots, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=train_clean[numeric_cols], ax=ax, palette="Set2")
    ax.set_title("Boxplots des colonnes numériques (train_clean)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
    plt.tight_layout()
    results["plot_boxplots_train_clean"] = fig_boxplots

    # Étape 7 : Visualisation de la distribution de la cible (target_col) dans train_clean
    fig_distribution, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=train_clean[target_col], bins=150, kde=True, ax=ax)
    ax.set_title(f'Distribution de la cible ({target_col}) dans train_clean')
    ax.set_xlim(0, 20000)
    ax.set_xticklabels(ax.get_xticks(), rotation=45, fontsize=8)
    plt.tight_layout()
    results["plot_distribution_target_train_clean"] = fig_distribution

    return results