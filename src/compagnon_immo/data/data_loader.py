"""
Module de chargement des données.
"""

import pandas as pd
from typing import Optional, Dict, Any
import logging


class DataLoader:
    """Classe pour charger les données depuis différents formats."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Charger des données depuis un fichier CSV.

        Args:
            file_path (str): Chemin vers le fichier CSV
            **kwargs: Arguments supplémentaires pour pd.read_csv

        Returns:
            pd.DataFrame: Données chargées
        """
        try:
            # Paramètres par défaut
            default_kwargs = {
                'sep': ';',
                'encoding': 'utf-8',
                'low_memory': False
            }
            default_kwargs.update(kwargs)

            df = pd.read_csv(file_path, **default_kwargs)
            self.logger.info(f"Données chargées depuis {file_path}: {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {file_path}: {e}")
            raise

    def load_excel(self, file_path: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """
        Charger des données depuis un fichier Excel.

        Args:
            file_path (str): Chemin vers le fichier Excel
            sheet_name (str): Nom de la feuille
            **kwargs: Arguments supplémentaires pour pd.read_excel

        Returns:
            pd.DataFrame: Données chargées
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            self.logger.info(f"Données chargées depuis {file_path} (feuille: {sheet_name}): {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {file_path}: {e}")
            raise

    def load_data(self, file_path: str, file_format: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Charger des données selon le format détecté automatiquement.

        Args:
            file_path (str): Chemin vers le fichier
            file_format (str, optional): Format du fichier ('csv', 'excel', etc.)
            **kwargs: Arguments supplémentaires

        Returns:
            pd.DataFrame: Données chargées
        """
        if file_format is None:
            if file_path.endswith('.csv'):
                file_format = 'csv'
            elif file_path.endswith(('.xlsx', '.xls')):
                file_format = 'excel'
            else:
                raise ValueError(f"Format de fichier non supporté: {file_path}")

        if file_format == 'csv':
            return self.load_csv(file_path, **kwargs)
        elif file_format == 'excel':
            return self.load_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Format non supporté: {file_format}")
