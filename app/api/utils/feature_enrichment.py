"""
Module pour l'enrichissement des features.
"""

import pandas as pd
import os
import numpy as np
from typing import Dict, Any, Optional

# Chemin vers le fichier CSV contenant les données d'enrichissement
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    '..', '..', 'df_sales_clean_with_cluster.csv'
)

# Chargement du CSV en mémoire au démarrage
if os.path.exists(CSV_PATH):
    df_features = pd.read_csv(CSV_PATH, sep=';', dtype=str)
else:
    df_features = pd.DataFrame()


def enrich_features_from_code_postal(code_postal: str) -> Dict[str, Any]:
    """
    Enrichit les features à partir du code postal.

    Args:
        code_postal (str): Code postal pour lequel enrichir les données.

    Returns:
        dict: Dictionnaire contenant les features enrichies.
    """
    if df_features.empty:
        # Si le fichier CSV n'est pas chargé, retourner des valeurs par défaut
        return {
            "x": 2.3522,  # Coordonnées par défaut (ex. Paris)
            "y": 48.8566,
            "cluster": 0,
            "dpeL": "D"
        }

    # Filtrer les données par code postal
    enriched_data = df_features[df_features["code_postal"] == code_postal]

    if enriched_data.empty:
        # Si aucun résultat, retourner des valeurs par défaut
        return {
            "x": 2.3522,
            "y": 48.8566,
            "cluster": 0,
            "dpeL": "D"
        }

    # Retourner les données enrichies
    return {
        "x": float(enriched_data["x"].iloc[0]),
        "y": float(enriched_data["y"].iloc[0]),
        "cluster": int(enriched_data["cluster"].iloc[0]),
        "dpeL": enriched_data["dpeL"].iloc[0]
    }


def harmonize_features(X_transformed: Any, feature_names: list) -> pd.DataFrame:
    """
    Harmonise un DataFrame ou un array numpy pour qu'il ait exactement les colonnes attendues,
    dans le bon ordre, en ajoutant les colonnes manquantes (remplies de 0) et en supprimant les colonnes en trop.

    Args:
        X_transformed (pd.DataFrame | np.ndarray): Données transformées.
        feature_names (list): Liste des noms de colonnes attendues.

    Returns:
        pd.DataFrame: Données harmonisées avec les colonnes dans le bon ordre.

    Raises:
        ValueError: Si le pipeline ne fournit pas les bons noms de colonnes.
    """
    # Si X_transformed est un array numpy, on le convertit en DataFrame
    if not isinstance(X_transformed, pd.DataFrame):
        if X_transformed.shape[1] == len(feature_names):
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        else:
            raise ValueError(
                "Impossible d'harmoniser : le pipeline ne fournit pas les bons noms de colonnes."
            )

    # Ajoute les colonnes manquantes
    for col in feature_names:
        if col not in X_transformed.columns:
            X_transformed[col] = 0  # ou np.nan selon votre choix

    # Supprime les colonnes en trop et réordonne
    X_transformed = X_transformed[feature_names]
    return X_transformed