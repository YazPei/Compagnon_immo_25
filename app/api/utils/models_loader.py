"""
Module pour charger les modèles et les préprocesseurs.
"""

import os
import joblib
import numpy as np
from typing import Optional

def get_model(version: Optional[str] = None):
    """
    Charge un modèle spécifique en fonction de la version.

    Args:
        version (str, optional): Version du modèle à charger. Si None, charge la dernière version.

    Returns:
        Modèle chargé.
    """
    model_dir = os.getenv("MODEL_DIR", "models/")
    version = version or "latest"
    model_path = os.path.join(model_dir, f"model_{version}.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    return joblib.load(model_path)

def get_preprocessor():
    """Retourne un préprocesseur mock pour les tests."""
    class MockPreprocessor:
        @property
        def feature_names_in_(self):
            return [
                "surface", "nb_pieces", "nb_chambres", "etage", 
                "annee_construction", "ascenseur", "balcon", "terrasse",
                "parking", "cave", "x", "y", "cluster", "dpeL",
                "type_vente", "type_appartement", "type_maison", 
                "type_studio", "type_loft"
            ]
        
        def transform(self, X):
            # Si X est un DataFrame pandas
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = np.array(X)
            
            # S'assurer que c'est 2D
            if X_array.ndim == 1:
                X_array = X_array.reshape(1, -1)
            
            # Normalisation mock (retourner les mêmes valeurs)
            return X_array
    
    return MockPreprocessor()

def get_model_metadata(version: Optional[str] = None):
    """
    Retourne les métadonnées du modèle.

    Args:
        version (str, optional): Version du modèle. Si None, retourne les métadonnées de la dernière version.

    Returns:
        dict: Métadonnées du modèle.
    """
    return {
        "feature_names": [
            "surface", "nb_pieces", "nb_chambres", "etage",
            "annee_construction", "ascenseur", "balcon", "terrasse",
            "parking", "cave", "x", "y", "cluster", "dpeL",
            "type_vente", "type_appartement", "type_maison",
            "type_studio", "type_loft"
        ],
        "version": version or "latest",
        "model_type": "RandomForestRegressor"
    }