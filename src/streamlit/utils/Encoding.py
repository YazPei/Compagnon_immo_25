################################################################################
#                                                                              #
#                          ENCODAGE ET TRANSFORMATION                          #
#                                                                              #
################################################################################
"""
Module contenant les fonctions d'encodage et de transformation des variables
pour un projet d'analyse immobilière basé sur Streamlit, pandas et scikit-learn.

Ce module regroupe toutes les fonctions permettant de transformer les données brutes
en features exploitables par les algorithmes d'apprentissage automatique.
"""

# Standard library
import os
import warnings
from typing import List, Dict, Tuple, Union, Optional

# Data manipulation
import numpy as np
import pandas as pd

# Geospatial
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    warnings.warn("Les modules geopandas et shapely ne sont pas installés. "
                 "Les fonctionnalités géospatiales seront limitées.")

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# Clustering
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN_CLUSTER = True
except ImportError:
    HAS_SKLEARN_CLUSTER = False
    warnings.warn("Le module sklearn.cluster n'est pas installé. "
                 "Les fonctionnalités de clustering ne seront pas disponibles.")

try:
    from category_encoders import TargetEncoder
    HAS_CATEGORY_ENCODERS = True
except ImportError:
    HAS_CATEGORY_ENCODERS = False
    warnings.warn("Le module category_encoders n'est pas installé. "
                 "L'encodage par cible ne sera pas disponible.")


################################################################################
########################## DÉFINITION DES COLONNES #############################
################################################################################

def get_column_groups() -> Dict[str, List[str]]:
    """
    Retourne un dictionnaire contenant les groupes de colonnes par type d'encodage.
    
    Returns:
        Dict[str, List[str]]: Dictionnaire avec les groupes de colonnes
    """
    column_groups = {
        "ordinal_cols": [
            "ges_class", "dpeL", "logement_neuf", "nb_pieces", 
            "bain", "eau", "nb_toilettes", "balcon"
        ],
        "onehot_cols": [
            "typedebien", "typedetransaction", "chauffage_mode", "chauffage_energie"
        ],
        "target_cols": [
            "etage", "nb_etages", "exposition", "chauffage_energie", "chauffage_systeme"
        ],
        "numeric_cols": [
            "surface", "surface_terrain", "dpeC", "places_parking", "charges_copro",
            "loyer_m2_median_n6", "nb_log_n6", "taux_rendement_n6",
            "loyer_m2_median_n7", "nb_log_n7", "taux_rendement_n7"
        ],
        "geo_cols": ["x_geo", "y_geo", "z_geo"],
        "year_col": ["annee_construction"],
        "year_order": [
            "après 2021", "2013-2021", "2006-2012", "2001-2005",
            "1989-2000", "1983-1988", "1978-1982", "1975-1977",
            "1948-1974", "avant 1948"
        ]
    }
    return column_groups


def get_all_feature_columns() -> List[str]:
    """
    Retourne la liste complète des colonnes à utiliser comme features,
    en combinant tous les groupes de colonnes définis.
    
    Returns:
        List[str]: Liste de toutes les colonnes features
    """
    column_groups = get_column_groups()
    features = (
        column_groups["ordinal_cols"] 
        + column_groups["onehot_cols"] 
        + column_groups["target_cols"] 
        + column_groups["numeric_cols"] 
        + column_groups["geo_cols"] 
        + column_groups["year_col"] 
        + ["date"]  # On inclut la date pour l'encodage cyclique
    )
    return features


def get_numeric_columns(data: pd.DataFrame, group_col: str, excluded_cols: Optional[List[str]] = None) -> List[str]:
    """
    Retourne les colonnes numériques en excluant une colonne de regroupement et d'autres colonnes spécifiées.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données
        group_col (str): Colonne de regroupement à exclure
        excluded_cols (Optional[List[str]]): Liste de colonnes supplémentaires à exclure
        
    Returns:
        List[str]: Liste des colonnes numériques filtrées
    """
    if excluded_cols is None:
        excluded_cols = ['mapCoordonneesLatitude', 'mapCoordonneesLongitude']
    return [
        col for col in data.select_dtypes(include='number').columns
        if col != group_col and col not in excluded_cols
    ]


################################################################################
########################## ENCODAGE DES DATES ##################################
################################################################################

def cyclical_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique un encodage cyclique aux dates pour capturer leur nature périodique.
    
    Cette fonction transforme la colonne 'date' en extrayant le mois, le jour de la semaine
    et l'heure, puis applique une transformation sinusoïdale et cosinusoïdale pour
    préserver la nature cyclique de ces variables temporelles.
    
    Args:
        df (pd.DataFrame): DataFrame contenant une colonne 'date'
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles colonnes d'encodage cyclique
                     et sans la colonne 'date' d'origine
    """
    df = df.copy()
    
    # Assurer que 'date' est au format datetime64
    df["date"] = pd.to_datetime(df["date"])
    
    # Extraire les composantes temporelles
    df["month"] = df["date"].dt.month      # 1–12
    df["dow"]   = df["date"].dt.weekday    # 0–6
    df["hour"]  = df["date"].dt.hour       # 0–23

    # Appliquer les transformations sin/cos pour chaque périodicité
    for col, period in [("month", 12), ("dow", 7), ("hour", 24)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
        df.drop(columns=[col], inplace=True)

    # Supprimer la colonne brute 'date' qui n'est plus nécessaire
    df.drop(columns=["date"], inplace=True)
    
    return df


def extract_date_components(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Extrait les composantes d'une colonne date (année, mois, jour, etc.)
    sans appliquer d'encodage cyclique.
    
    Args:
        df (pd.DataFrame): DataFrame contenant une colonne date
        date_col (str): Nom de la colonne date à traiter
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles colonnes de composantes de date
    """
    df = df.copy()
    
    # Assurer que la colonne date est au format datetime64
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extraire les composantes temporelles
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["quarter"] = df[date_col].dt.quarter
    
    return df


################################################################################
####################### TRANSFORMATION GÉOGRAPHIQUE ############################
################################################################################

def transform_coordinates_to_cartesian(df: pd.DataFrame, 
                                      lat_col: str = "mapCoordonneesLatitude", 
                                      lon_col: str = "mapCoordonneesLongitude") -> pd.DataFrame:
    """
    Transforme les coordonnées géographiques (latitude/longitude) en coordonnées cartésiennes 3D.
    
    Cette transformation présente plusieurs avantages:
    1. Elle préserve la proximité géographique dans l'espace euclidien
    2. Elle évite les problèmes liés à la discontinuité des coordonnées
    3. Elle permet aux algorithmes d'apprentissage de mieux capturer les relations spatiales
    
    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes de latitude et longitude
        lat_col (str): Nom de la colonne de latitude
        lon_col (str): Nom de la colonne de longitude
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles colonnes x_geo, y_geo, z_geo
    """
    df = df.copy()
    
    # Conversion des coordonnées en radians
    lat_rad = np.radians(df[lat_col])
    lon_rad = np.radians(df[lon_col])
    
    # Calcul des coordonnées cartésiennes
    df['x_geo'] = np.cos(lat_rad) * np.cos(lon_rad)
    df['y_geo'] = np.cos(lat_rad) * np.sin(lon_rad)
    df['z_geo'] = np.sin(lat_rad)
    
    return df


def assign_postal_codes(df: pd.DataFrame, 
                       geo_file_path: str,
                       lat_col: str = "mapCoordonneesLatitude", 
                       lon_col: str = "mapCoordonneesLongitude") -> pd.DataFrame:
    """
    Attribue des codes postaux aux points géographiques en utilisant un fichier GeoJSON.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes de latitude et longitude
        geo_file_path (str): Chemin vers le fichier GeoJSON des contours postaux
        lat_col (str): Nom de la colonne de latitude
        lon_col (str): Nom de la colonne de longitude
        
    Returns:
        pd.DataFrame: DataFrame avec la nouvelle colonne 'codePostal'
        
    Raises:
        ImportError: Si geopandas n'est pas installé
    """
    if not HAS_GEO:
        raise ImportError("Cette fonction nécessite geopandas et shapely. "
                         "Installez-les avec 'pip install geopandas shapely'.")
    
    df = df.copy()
    
    # Chargement du fichier GeoJSON
    pcodes = gpd.read_file(geo_file_path)
    
    # Création du GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

    # Jointure spatiale
    result = gpd.sjoin(gdf, pcodes[['codePostal', 'geometry']], how='left', predicate='within')

    # Conservation du premier codePostal trouvé pour chaque point
    result_unique = result[['codePostal']].groupby(result.index).first()

    # Alignement sur l'index du DataFrame d'origine
    df['codePostal'] = result_unique['codePostal'].astype(str)
    
    return df


################################################################################
####################### ENCODAGE DES VARIABLES CATÉGORIELLES ###################
################################################################################

def encode_ordinal_variables(df: pd.DataFrame, ordinal_cols: List[str], 
                           categories_dict: Optional[Dict[str, List]] = None) -> pd.DataFrame:
    """
    Encode les variables ordinales selon un ordre spécifié.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les variables à encoder
        ordinal_cols (List[str]): Liste des colonnes à encoder de façon ordinale
        categories_dict (Optional[Dict[str, List]]): Dictionnaire des catégories ordonnées par colonne
        
    Returns:
        pd.DataFrame: DataFrame avec les variables ordinales encodées
    """
    df = df.copy()
    
    # Dictionnaire par défaut pour certaines variables
    default_categories = {
        "ges_class": ["A", "B", "C", "D", "E", "F", "G"],
        "dpeL": ["A", "B", "C", "D", "E", "F", "G"],
        "logement_neuf": ["oui", "non"],
        "annee_construction": [
            "après 2021", "2013-2021", "2006-2012", "2001-2005",
            "1989-2000", "1983-1988", "1978-1982", "1975-1977",
            "1948-1974", "avant 1948"
        ]
    }
    
    # Utiliser les catégories fournies ou les valeurs par défaut
    if categories_dict is None:
        categories_dict = default_categories
    
    # Encoder chaque variable ordinale
    for col in ordinal_cols:
        if col in categories_dict:
            # Utiliser l'ordre spécifié
            categories = categories_dict[col]
            encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = encoder.fit_transform(df[[col]])
        else:
            # Utiliser l'ordre naturel des valeurs uniques
            unique_values = sorted(df[col].dropna().unique())
            encoder = OrdinalEncoder(categories=[unique_values], handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = encoder.fit_transform(df[[col]])
    
    return df


def encode_onehot_variables(df: pd.DataFrame, onehot_cols: List[str], 
                          drop: str = 'first', handle_unknown: str = 'ignore') -> pd.DataFrame:
    """
    Applique un encodage one-hot aux variables catégorielles spécifiées.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les variables à encoder
        onehot_cols (List[str]): Liste des colonnes à encoder avec one-hot
        drop (str): Stratégie de suppression ('first' ou None)
        handle_unknown (str): Stratégie pour les valeurs inconnues
        
    Returns:
        pd.DataFrame: DataFrame avec les variables encodées en one-hot
    """
    df = df.copy()
    
    # Créer l'encodeur
    encoder = OneHotEncoder(sparse_output=False, drop=drop, handle_unknown=handle_unknown)
    
    # Appliquer l'encodage
    for col in onehot_cols:
        # Ajuster et transformer
        encoded = encoder.fit_transform(df[[col]])
        
        # Obtenir les noms des nouvelles colonnes
        categories = encoder.categories_[0]
        if drop == 'first':
            categories = categories[1:]
        
        # Créer les noms des colonnes
        col_names = [f"{col}_{cat}" for cat in categories]
        
        # Ajouter les colonnes encodées au DataFrame
        encoded_df = pd.DataFrame(encoded, index=df.index, columns=col_names)
        df = pd.concat([df, encoded_df], axis=1)
        
        # Supprimer la colonne originale
        df.drop(columns=[col], inplace=True)
    
    return df


def encode_target_variables(df: pd.DataFrame, target_cols: List[str], 
                          target_variable: str) -> pd.DataFrame:
    """
    Applique un encodage par cible (target encoding) aux variables spécifiées.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les variables à encoder
        target_cols (List[str]): Liste des colonnes à encoder par cible
        target_variable (str): Nom de la variable cible
        
    Returns:
        pd.DataFrame: DataFrame avec les variables encodées par cible
        
    Raises:
        ImportError: Si category_encoders n'est pas installé
    """
    if not HAS_CATEGORY_ENCODERS:
        raise ImportError("Cette fonction nécessite category_encoders. "
                         "Installez-le avec 'pip install category_encoders'.")
    
    df = df.copy()
    
    # Créer l'encodeur
    encoder = TargetEncoder()
    
    # Appliquer l'encodage pour chaque colonne
    for col in target_cols:
        if col in df.columns:
            # Ajuster l'encodeur sur les données non-nulles
            mask = df[col].notna() & df[target_variable].notna()
            if mask.sum() > 0:
                encoded_values = encoder.fit_transform(df.loc[mask, col], df.loc[mask, target_variable])
                df.loc[mask, f"{col}_encoded"] = encoded_values
                
                # Gérer les valeurs manquantes avec la moyenne
                if (~mask).sum() > 0:
                    df.loc[~mask, f"{col}_encoded"] = encoded_values.mean()
                
                # Supprimer la colonne originale si demandé
                df.drop(columns=[col], inplace=True)
    
    return df


################################################################################
####################### TRANSFORMATION DES VARIABLES BOOLÉENNES ################
################################################################################

def encode_boolean_variables(df: pd.DataFrame, bool_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convertit les variables booléennes en entiers (0/1).
    
    Args:
        df (pd.DataFrame): DataFrame contenant les variables à encoder
        bool_cols (Optional[List[str]]): Liste des colonnes booléennes à encoder
                                        Si None, détecte automatiquement les colonnes de type bool
        
    Returns:
        pd.DataFrame: DataFrame avec les variables booléennes encodées en entiers
    """
    df = df.copy()
    
    # Si aucune liste n'est fournie, détecter automatiquement les colonnes booléennes
    if bool_cols is None:
        bool_cols = df.select_dtypes(include='bool').columns.tolist()
    
    # Convertir chaque colonne booléenne en entier
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df


################################################################################
####################### CRÉATION DE PIPELINE D'ENCODAGE #######################
################################################################################

def create_encoding_pipeline(ordinal_cols: List[str], onehot_cols: List[str], 
                           numeric_cols: List[str], has_date: bool = True,
                           has_geo: bool = True) -> Pipeline:
    """
    Crée un pipeline complet d'encodage et de transformation des variables.
    
    Args:
        ordinal_cols (List[str]): Colonnes à encoder de façon ordinale
        onehot_cols (List[str]): Colonnes à encoder avec one-hot
        numeric_cols (List[str]): Colonnes numériques à standardiser
        has_date (bool): Indique si le pipeline doit inclure l'encodage des dates
        has_geo (bool): Indique si le pipeline doit inclure la transformation géographique
        
    Returns:
        Pipeline: Pipeline scikit-learn complet pour l'encodage des données
    """
    # Transformateurs pour chaque type de variable
    transformers = []
    
    # Encodage ordinal
    if ordinal_cols:
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        transformers.append(('ordinal', ordinal_transformer, ordinal_cols))
    
    # Encodage one-hot
    if onehot_cols:
        onehot_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(('onehot', onehot_transformer, onehot_cols))
    
    # Standardisation des variables numériques
    if numeric_cols:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_transformer, numeric_cols))
    
    # Encodage cyclique des dates
    if has_date:
        date_transformer = FunctionTransformer(cyclical_encode)
        transformers.append(('date', date_transformer, ['date']))
    
    # Transformation des coordonnées géographiques
    if has_geo:
        geo_transformer = FunctionTransformer(
            transform_coordinates_to_cartesian,
            kw_args={'lat_col': 'mapCoordonneesLatitude', 'lon_col': 'mapCoordonneesLongitude'}
        )
        transformers.append(('geo', geo_transformer, ['mapCoordonneesLatitude', 'mapCoordonneesLongitude']))
    
    # Création du ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Ignorer les colonnes non spécifiées
    )
    
    # Création du pipeline complet
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return pipeline


################################################################################
####################### FONCTIONS UTILITAIRES #################################
################################################################################

def regroup_cp(code_postal: str) -> str:
    """
    Regroupe les codes postaux en zones mixtes.
    
    Args:
        code_postal (str): Code postal à regrouper
        
    Returns:
        str: Zone mixte correspondante
    """
    try:
        # Extraire les 2 premiers chiffres (département)
        dept = code_postal[:2]
        
        # Cas particuliers pour Paris, Lyon, Marseille
        if dept == "75":
            return "75000"  # Paris
        elif dept == "69" and code_postal.startswith("69"):
            return "69000"  # Lyon
        elif dept == "13" and code_postal.startswith("130"):
            return "13000"  # Marseille
        else:
            # Pour les autres, on garde les 3 premiers chiffres
            return code_postal[:3] + "00"
    except:
        return "00000"  # Valeur par défaut en cas d'erreur


def get_code_postal_final(zone_mixte: str) -> str:
    """
    Convertit une zone mixte en code postal final.
    
    Args:
        zone_mixte (str): Zone mixte à convertir
        
    Returns:
        str: Code postal final
    """
    # Cas particuliers
    if zone_mixte == "75000":
        return "75000"  # Paris
    elif zone_mixte == "69000":
        return "69000"  # Lyon
    elif zone_mixte == "13000":
        return "13000"  # Marseille
    else:
        # Pour les autres, on ajoute des zéros
        return zone_mixte.ljust(5, '0')


def create_year_bins(df: pd.DataFrame, year_col: str = "annee_construction") -> pd.DataFrame:
    """
    Convertit les années de construction en catégories ordinales.
    
    Args:
        df (pd.DataFrame): DataFrame contenant la colonne d'année
        year_col (str): Nom de la colonne contenant l'année de construction
        
    Returns:
        pd.DataFrame: DataFrame avec la colonne d'année convertie en catégories
    """
    df = df.copy()
    
    # Définition des intervalles
    bins = [0, 1948, 1975, 1978, 1983, 1989, 2001, 2006, 2013, 2021, float('inf')]
    labels = [
        "avant 1948", "1948-1974", "1975-1977", "1978-1982", 
        "1983-1988", "1989-2000", "2001-2005", "2006-2012", 
        "2013-2021", "après 2021"
    ]
    
    # Conversion de la colonne en numérique
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    
    # Application des bins
    df[year_col] = pd.cut(
        df[year_col], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    return df


################################################################################
####################### FONCTIONS DE CLUSTERING ###############################
################################################################################

def prepare_data_for_clustering(df: pd.DataFrame, 
                              date_col: str = "date", 
                              code_postal_col: str = "codePostal",
                              price_col: str = "prix_m2_vente") -> pd.DataFrame:
    """
    Prépare les données pour le clustering en ajoutant des colonnes dérivées.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données immobilières
        date_col (str): Nom de la colonne de date
        code_postal_col (str): Nom de la colonne de code postal
        price_col (str): Nom de la colonne de prix au m²
        
    Returns:
        pd.DataFrame: DataFrame avec les colonnes préparées pour le clustering
    """
    df = df.copy()
    
    # Conversion de la date en datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extraction de l'année et du mois
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    
    # Création des zones mixtes et codes postaux reconstruits
    df["zone_mixte"] = df[code_postal_col].astype(str).apply(regroup_cp)
    df["codePostal_recons"] = df["zone_mixte"].apply(get_code_postal_final)
    
    # Tri par zone et date pour les calculs de lag et rolling
    df.sort_values(["zone_mixte", date_col], inplace=True)
    
    # Calcul du lag de prix (mois précédent)
    df[f"{price_col}_lag_1m"] = df.groupby("zone_mixte")[price_col].shift(1)
    
    # Calcul de la moyenne mobile sur 3 mois
    df[f"{price_col}_roll_3m"] = (
        df.groupby("zone_mixte")[price_col]
          .rolling(3, closed="left")
          .mean()
          .reset_index(level=0, drop=True)
    )
    
    return df


def calculate_tcam(df: pd.DataFrame, 
                  group_col: str = "codePostal_recons", 
                  date_col: str = "date",
                  price_col: str = "prix_m2_vente") -> pd.DataFrame:
    """
    Calcule le Taux de Croissance Annuel Moyen (TCAM) par groupe.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données immobilières
        group_col (str): Colonne de regroupement (ex: code postal)
        date_col (str): Nom de la colonne de date
        price_col (str): Nom de la colonne de prix
        
    Returns:
        pd.DataFrame: DataFrame avec le TCAM par groupe
    """
    # Préparation des données
    df_prep = df.copy()
    df_prep[date_col] = pd.to_datetime(df_prep[date_col])
    
    # Tri par groupe et date
    df_prep.sort_values([group_col, date_col], inplace=True)
    
    # Calcul du TCAM
    tcam_df = (
        df_prep
        .groupby(group_col)[price_col]
        .apply(lambda s: (s.iloc[-1] / s.iloc[0]) ** (12 / max(len(s), 1)) - 1 if len(s) > 1 else 0)
        .rename("tc_am_reg")
        .reset_index()
    )
    
    return tcam_df


def aggregate_monthly_data(df: pd.DataFrame, 
                         group_cols: List[str] = ["Year", "Month", "zone_mixte"],
                         price_col: str = "prix_m2_vente",
                         lag_col: str = "prix_m2_vente_lag_1m",
                         roll_col: str = "prix_m2_vente_roll_3m") -> pd.DataFrame:
    """
    Agrège les données immobilières par mois et par zone.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données immobilières préparées
        group_cols (List[str]): Colonnes de regroupement
        price_col (str): Nom de la colonne de prix
        lag_col (str): Nom de la colonne de lag de prix
        roll_col (str): Nom de la colonne de moyenne mobile
        
    Returns:
        pd.DataFrame: DataFrame agrégé par mois et par zone
    """
    # Agrégation mensuelle
    monthly_df = (
        df
        .groupby(group_cols)
        .agg(
            prix_m2_vente=(price_col, "mean"),
            volume_ventes=(price_col, "count"),
            avg_lag_1m=(lag_col, "mean"),
            avg_roll_3m=(roll_col, "mean"),
            prix_m2_std=(price_col, "std"),
            prix_m2_min=(price_col, "min"),
            prix_m2_max=(price_col, "max")
        )
        .reset_index()
    )
    
    # Calcul du coefficient de variation
    monthly_df["prix_m2_cv"] = monthly_df["prix_m2_std"] / monthly_df["prix_m2_vente"]
    
    return monthly_df


def create_clustering_features(df: pd.DataFrame, 
                             group_col: str = "zone_mixte",
                             price_col: str = "prix_m2_vente") -> pd.DataFrame:
    """
    Crée les features pour le clustering à partir des données agrégées.
    
    Args:
        df (pd.DataFrame): DataFrame agrégé par mois et par zone
        group_col (str): Colonne de regroupement (zone)
        price_col (str): Nom de la colonne de prix
        
    Returns:
        pd.DataFrame: DataFrame avec une ligne par zone et les features pour le clustering
    """
    # Création du DataFrame pour le clustering
    cluster_input = (
        df
        .groupby(group_col)
        .agg(
            # Prix moyen et volatilité
            prix_moyen=(price_col, "mean"),
            prix_std=(price_col, "std"),
            prix_cv=("prix_m2_cv", "mean"),
            
            # Dynamique du marché
            volume_moyen=("volume_ventes", "mean"),
            volume_std=("volume_ventes", "std"),
            
            # Tendance et saisonnalité
            prix_min=(price_col, "min"),
            prix_max=(price_col, "max"),
            amplitude=lambda x: x[price_col].max() - x[price_col].min(),
            
            # Lag et tendance
            lag_ratio=lambda x: (x[price_col] / x["avg_lag_1m"]).mean(),
            roll_ratio=lambda x: (x[price_col] / x["avg_roll_3m"]).mean()
        )
        .reset_index()
    )
    
    # Calcul de l'amplitude relative
    cluster_input["amplitude_relative"] = cluster_input["amplitude"] / cluster_input["prix_moyen"]
    
    return cluster_input


def perform_clustering(df: pd.DataFrame, 
                     n_clusters: int = 5, 
                     random_state: int = 42,
                     features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Effectue un clustering K-means sur les données préparées.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les features pour le clustering
        n_clusters (int): Nombre de clusters à créer
        random_state (int): Graine aléatoire pour la reproductibilité
        features (Optional[List[str]]): Liste des colonnes à utiliser pour le clustering
                                       Si None, utilise toutes les colonnes numériques
        
    Returns:
        pd.DataFrame: DataFrame original avec une colonne 'cluster' ajoutée
        
    Raises:
        ImportError: Si sklearn.cluster n'est pas installé
    """
    if not HAS_SKLEARN_CLUSTER:
        raise ImportError("Cette fonction nécessite sklearn.cluster. "
                         "Installez-le avec 'pip install scikit-learn'.")
    
    df = df.copy()
    
    # Si aucune liste de features n'est fournie, utiliser toutes les colonnes numériques
    # sauf la première colonne (généralement l'identifiant de zone)
    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()
        if len(features) > 0 and features[0] == df.columns[0]:
            features = features[1:]
    
    # Standardisation des features
    scaler = StandardScaler()
    X = df[features].fillna(df[features].mean())
    X_scaled = scaler.fit_transform(X)
    
    # Application de K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df


def apply_clustering_pipeline(df: pd.DataFrame, 
                            n_clusters: int = 5,
                            date_col: str = "date",
                            code_postal_col: str = "codePostal",
                            price_col: str = "prix_m2_vente") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applique le pipeline complet de clustering aux données immobilières.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données immobilières brutes
        n_clusters (int): Nombre de clusters à créer
        date_col (str): Nom de la colonne de date
        code_postal_col (str): Nom de la colonne de code postal
        price_col (str): Nom de la colonne de prix au m²
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame original avec les colonnes de clustering ajoutées
            - DataFrame des clusters avec leurs caractéristiques
    """
    # 1. Préparation des données
    df_prepared = prepare_data_for_clustering(
        df, 
        date_col=date_col, 
        code_postal_col=code_postal_col,
        price_col=price_col
    )
    
    # 2. Calcul du TCAM
    tcam_df = calculate_tcam(
        df_prepared, 
        group_col="codePostal_recons", 
        date_col=date_col,
        price_col=price_col
    )
    
    # 3. Agrégation mensuelle
    monthly_df = aggregate_monthly_data(
        df_prepared,
        group_cols=["Year", "Month", "zone_mixte"],
        price_col=price_col,
        lag_col=f"{price_col}_lag_1m",
        roll_col=f"{price_col}_roll_3m"
    )
    
    # 4. Création des features pour le clustering
    cluster_input = create_clustering_features(
        monthly_df,
        group_col="zone_mixte",
        price_col="prix_m2_vente"
    )
    
    # 5. Fusion avec le TCAM
    cluster_input = cluster_input.merge(
        tcam_df,
        left_on="zone_mixte",
        right_on="codePostal_recons",
        how="left"
    )
    
    # 6. Clustering
    cluster_result = perform_clustering(
        cluster_input,
        n_clusters=n_clusters,
        random_state=42
    )
    
    # 7. Ajout des clusters au DataFrame original
    zone_cluster_map = dict(zip(cluster_result["zone_mixte"], cluster_result["cluster"]))
    df_prepared["cluster"] = df_prepared["zone_mixte"].map(zone_cluster_map)
    
    return df_prepared, cluster_result
