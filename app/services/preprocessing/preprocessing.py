"""
Module pour le prétraitement des données.
"""

import os
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime
import click

run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")


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
@click.option("--input-path", type=click.Path(exists=True), prompt="Chemin du fichier d'entrée")
@click.option("--output-path", type=click.Path(), prompt="Chemin du dossier de sortie")
def main(input_path, output_path):
    """
    Commande CLI pour exécuter le pipeline de prétraitement.
    """
    preprocessing_pipeline(input_path, output_path)


if __name__ == "__main__":
    main()
