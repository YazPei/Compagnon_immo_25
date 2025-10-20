"""
Module contenant la classe PriceModel pour la prédiction des prix immobiliers.
"""

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from app.api.models.ml.custom_functions import replace_minus1

# Fonction custom utilisée lors de l'entraînement du modèle
# À adapter si tu as la vraie logique, sinon retourne x


def replace_minus1(x):
    return x


class PriceModel:
    """Classe pour gérer le modèle de prédiction des prix immobiliers."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise le modèle de prédiction.

        Args:
            model_path: Chemin vers le fichier .pkl du modèle
        """
        self.model: Optional[LGBMRegressor] = None
        self.feature_names: List[str] = []
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Charge le modèle depuis un fichier .pkl

        Args:
            model_path: Chemin vers le fichier .pkl

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le modèle n'est pas valide
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier modèle {model_path} n'existe pas")
        try:
            import sys

            from app.api.models.ml import custom_functions

            # Ajoute toutes les fonctions custom dans le scope global de __main__
            for func_name in dir(custom_functions):
                func = getattr(custom_functions, func_name)
                if callable(func) and not func_name.startswith("__"):
                    setattr(sys.modules["__main__"], func_name, func)
            self.model = joblib.load(model_path)
            # Récupérer les noms des features si disponibles
            if hasattr(self.model, "feature_name_"):
                self.feature_names = self.model.feature_name_
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du modèle: {str(e)}")

    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prépare les features pour la prédiction.

        Args:
            input_data: Dictionnaire contenant les données d'entrée

        Returns:
            DataFrame avec les features préparées
        """
        # TODO: Implémenter la préparation des features selon votre logique métier
        # Exemple basique:
        df = pd.DataFrame([input_data])

        # Assurez-vous que toutes les features nécessaires sont présentes
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Features manquantes: {missing_features}")

            # Réorganiser les colonnes dans le même ordre que lors de l'entraînement
            df = df[self.feature_names]

        return df

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Fait une prédiction de prix.

        Args:
            input_data: Dictionnaire contenant les données d'entrée

        Returns:
            Dict contenant la prédiction et l'intervalle de confiance

        Raises:
            ValueError: Si le modèle n'est pas chargé ou si les données sont invalides
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé")

        try:
            # Préparer les features
            X = self.prepare_features(input_data)

            # Faire la prédiction
            prediction = self.model.predict(X)[0]

            # TODO: Calculer l'intervalle de confiance selon votre méthode
            # Exemple simple avec ±10%
            confidence_interval = prediction * 0.10

            return {
                "estimation": float(prediction),
                "intervalle_confiance": {
                    "min": float(prediction - confidence_interval),
                    "max": float(prediction + confidence_interval),
                },
            }

        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction: {str(e)}")
