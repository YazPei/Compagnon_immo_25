import os
import time
from pathlib import Path
import click
import numpy as np
import pandas as pd
import polars as pl
import mlflow
from sklearn.neighbors import BallTree

def sample(input_file):
    chunks = pd.read_csv(input_file, sep=';', chunksize=100_000, on_bad_lines='skip', low_memory=False)
    df_sales_clean = pd.concat(chunks)
    df_sample = df_sales_clean.sample(frac=0.1, random_state=42)
    return df_sample

@click.command()
@click.option('--folder-path', type=click.Path(exists=True), prompt='📂 Chemin vers les données ventes')
@click.option('--output-folder', type=click.Path(), prompt='📁 Dossier de sortie')
def main(folder_path, output_folder):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("Import données")

    with mlflow.start_run(run_name="Import et sample"):
        input_file = os.path.join(folder_path, 'merged_sales_data.csv')
        df_sample = sample(input_file)

        # Tracking
        mlflow.log_param("output_folder", output_folder)

        # Export
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_folder) / "df_sample.csv"
        pl.from_pandas(df_sample).write_csv(output_path, separator=";")

        # Log artefact
        mlflow.log_artifact(str(output_path))
        print(f"✅ Export CSV terminé : {output_path}")

if __name__ == '__main__':
    main()

