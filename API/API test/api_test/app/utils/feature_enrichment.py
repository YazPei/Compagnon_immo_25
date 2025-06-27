import pandas as pd
import os
from typing import Dict, Any, Optional
import numpy as np

# Chargement du CSV en mémoire au démarrage
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'df_sales_clean_with_cluster.csv')

if os.path.exists(CSV_PATH):
    df_features = pd.read_csv(CSV_PATH, sep=';', dtype=str)
else:
    df_features = pd.DataFrame()

def enrich_features_from_code_postal(code_postal: str) -> Dict[str, Any]:
    """
    Retourne un dictionnaire de features enrichies à partir du code postal.
    Si aucune correspondance, retourne un dict vide.
    """
    if df_features.empty:
        return {}
    row = df_features[df_features['codePostal'] == str(code_postal)]
    if row.empty:
        return {}
    # On retourne la première ligne trouvée sous forme de dict
    return row.iloc[0].to_dict()

def harmonize_features(X_transformed, feature_names):
    """
    Harmonise un DataFrame ou un array numpy pour qu'il ait exactement les colonnes attendues,
    dans le bon ordre, en ajoutant les colonnes manquantes (remplies de 0) et en supprimant les colonnes en trop.
    """
    # Si X_transformed est un array numpy, on le convertit en DataFrame
    if not isinstance(X_transformed, pd.DataFrame):
        # Si X_transformed a déjà le bon nombre de colonnes, on suppose que l'ordre est correct
        if X_transformed.shape[1] == len(feature_names):
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        else:
            # Sinon, on ne peut pas deviner les noms : il faut les récupérer du pipeline
            raise ValueError("Impossible d'harmoniser : le pipeline ne fournit pas les bons noms de colonnes.")
    # Si DataFrame mais mauvais nombre de colonnes, on force des noms génériques
    if len(X_transformed.columns) != len(X_transformed.iloc[0]):
        X_transformed.columns = [f"col_{i}" for i in range(X_transformed.shape[1])]
    # Ajoute les colonnes manquantes
    for col in feature_names:
        if col not in X_transformed.columns:
            X_transformed[col] = 0  # ou np.nan selon ton choix
    # Supprime les colonnes en trop et réordonne
    X_transformed = X_transformed[feature_names]
    return X_transformed 