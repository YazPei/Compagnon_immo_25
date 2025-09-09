"""Fonctions custom pour le modèle."""

import pandas as pd
import numpy as np


def replace_minus1(value):
    """
    Remplace -1 par 0.

    Args:
        value (int/float): Valeur à vérifier.

    Returns:
        int/float: 0 si la valeur est -1, sinon la valeur d'origine.
    """
    return 0 if value == -1 else value


def harmonize_features(X_transformed, feature_names):
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


def plus1(X):
    """
    Ajoute 1 à chaque élément.

    Args:
        X (int | float | np.ndarray | pd.Series): Valeur ou tableau.

    Returns:
        Même type que X : Valeur ou tableau avec 1 ajouté à chaque élément.
    """
    return X + 1


def invert11(X):
    """
    Inverse les valeurs entre 0 et 1 (1 - X).

    Args:
        X (int | float | np.ndarray | pd.Series): Valeur ou tableau.

    Returns:
        Même type que X : Valeur ou tableau inversé.
    """
    return 1 - X


def cyclical_encode(df):
    """
    Encode cycliquement les dates en utilisant des fonctions trigonométriques.

    Args:
        df (pd.DataFrame | pd.Series): Données contenant une colonne 'date' ou des dates.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes encodées cycliquement :
                      'month_sin', 'month_cos', 'dow_sin', 'dow_cos'.

    Raises:
        ValueError: Si la colonne 'date' est absente dans le DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        if "date" in df.columns:
            date = pd.to_datetime(df["date"])
        else:
            raise ValueError("La colonne 'date' est attendue pour l'encodage cyclique.")
    else:
        # Si c'est une Series
        date = pd.to_datetime(df)

    # Création des features cycliques
    res = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * date.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * date.dt.month / 12),
        "dow_sin": np.sin(2 * np.pi * date.dt.weekday / 7),
        "dow_cos": np.cos(2 * np.pi * date.dt.weekday / 7),
    })
    return res