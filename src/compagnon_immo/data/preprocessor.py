"""
Module pour le prétraitement des données.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import click
import mlflow
import pandas as pd

run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")


class DataPreprocessor:
    """Classe pour le prétraitement des données immobilières."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline de prétraitement des données.

        Args:
            df (pd.DataFrame): Données brutes

        Returns:
            pd.DataFrame: Données prétraitées
        """
        # Supprimer les doublons
        df_cleaned = df.drop_duplicates()
        print("Shape du Dataset après élimination des doublons :", df_cleaned.shape)

        return df_cleaned

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Sauvegarder les données prétraitées.

        Args:
            df (pd.DataFrame): Données à sauvegarder
            output_path (str): Chemin de sortie
        """
        output_file = Path(output_path) / f"cleaned_data_{run_suffix}.csv"
        df.to_csv(output_file, index=False)
        print(f"Données nettoyées sauvegardées dans : {output_file}")


def preprocessing_pipeline(input_path: str, output_path: str):
    """
    Pipeline de prétraitement des données.

    Args:
        input_path (str): Chemin vers le fichier d'entrée.
        output_path (str): Chemin vers le fichier de sortie.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("Preprocessing Données Immo")

    with mlflow.start_run(run_name="preprocessing_pipeline"):
        mlflow.set_tag("phase", "preprocessing")
        mlflow.set_tag("version", "v1.0")

        # Charger les données
        df = pd.read_csv(input_path, sep=";")
        print("Nombres de lignes en double :", df.duplicated().sum())

        # Supprimer les doublons
        df.drop_duplicates(inplace=True)
        print("Shape du Dataset après élimination des doublons :", df.shape)

        # Sauvegarder les données nettoyées
        output_file = Path(output_path) / f"cleaned_data_{run_suffix}.csv"
        df.to_csv(output_file, index=False)
        print(f"Données nettoyées sauvegardées dans : {output_file}")

        # Log dans MLflow
        mlflow.log_artifact(str(output_file), artifact_path="cleaned_data")


@click.command()
@click.option(
    "--input-path", type=click.Path(exists=True), prompt="Chemin du fichier d'entrée"
)
@click.option("--output-path", type=click.Path(), prompt="Chemin du dossier de sortie")
def main(input_path, output_path):
    """
    Commande CLI pour exécuter le pipeline de prétraitement.
    """
    preprocessing_pipeline(input_path, output_path)


if __name__ == "__main__":
    main()
