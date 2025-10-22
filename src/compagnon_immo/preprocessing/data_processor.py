"""
Module pour le prétraitement des données.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

class ImmoDataProcessor:
    """Classe pour le prétraitement des données immobilières."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Charger les données depuis un fichier."""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Données chargées depuis {file_path}, shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyer les données."""
        # Supprimer les doublons
        initial_shape = df.shape
        df = df.drop_duplicates()
        self.logger.info(f"Doublons supprimés: {initial_shape[0] - df.shape[0]} lignes")

        # Gérer les valeurs manquantes
        df = df.dropna()
        self.logger.info(f"Valeurs manquantes supprimées, shape finale: {df.shape}")

        return df

    def preprocess_features(self, df: pd.DataFrame, target_column: str = 'prix') -> Tuple[pd.DataFrame, pd.Series]:
        """Préparer les features et la target."""
        # Séparer features et target
        if target_column not in df.columns:
            raise ValueError(f"Colonne target '{target_column}' non trouvée")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encoder les variables catégorielles si nécessaire
        X = pd.get_dummies(X, drop_first=True)

        # Normaliser les features numériques
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            X[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])

        self.logger.info(f"Features préparées: {X.shape}, Target: {y.shape}")
        return X, y

    def save_processed_data(self, X: pd.DataFrame, y: pd.Series, output_path: str):
        """Sauvegarder les données traitées."""
        processed_df = X.copy()
        processed_df['target'] = y

        processed_df.to_csv(output_path, index=False)
        self.logger.info(f"Données traitées sauvegardées dans {output_path}")
