import os
import time
from pathlib import Path

import click
import mlflow
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# 🔒 Sécurisation immédiate AVANT tout appel à MLflow

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
else:
    ARTIFACT_DIR = os.path.abspath("mlruns")
    mlflow.set_tracking_uri("file://" + ARTIFACT_DIR)


experiment_name = "Import données"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(
        name=experiment_name, artifact_location="file://" + ARTIFACT_DIR
    )


def sample(input_file):
    chunks = pd.read_csv(
        input_file, sep=";", chunksize=100_000, on_bad_lines="skip", low_memory=False
    )
    df_sales_clean = pd.concat(chunks)
    df_sample = df_sales_clean.sample(frac=0.1, random_state=42)
    return df_sample


@click.command()
@click.option(
    "--folder-path",
    type=click.Path(exists=True),
    prompt="📂 Chemin vers les données ventes",
)
@click.option("--output-folder", type=click.Path(), prompt="📁 Dossier de sortie")
def main(folder_path, output_folder):
    with mlflow.start_run(run_name="Import et sample"):
        input_file = os.path.join(folder_path, "dvc_data.csv")
        df_sample = sample(input_file)

        # Tracking
        mlflow.log_param("output_folder", output_folder)

        # Export CSV
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_folder) / "df_sample.csv"
        cols_to_string = ["code_postal", "INSEE_COM", "departement", "commune"]
        for col in cols_to_string:
            if col in df_sample.columns:
                df_sample[col] = df_sample[col].astype(str)

        df_sample.to_csv(output_path, sep=";", index=False)

        # Log artefact (✅ maintenant dans mlruns/, pas /mlflow)
        mlflow.log_artifact(str(output_path))
        print(f"✅ Export CSV terminé : {output_path}")


if __name__ == "__main__":
    main()
