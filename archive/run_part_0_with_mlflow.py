
import mlflow
from pathlib import Path
import pandas as pd
import os

from part_0_preparation import *
from MLOPS.config_mlflow import setup_mlflow
import mlflow

# Initialise la config
setup_mlflow()


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("compagnon_immo")

with mlflow.start_run(run_name="Part-0_Preparation_donnees"):
    mlflow.log_param("source", "part0")
    # Log important config manuellement (exemple)
    mlflow.log_param("chunksize", 100000)
    mlflow.log_param("threads", min(4, os.cpu_count() or 1))
    #folder_path = '/home/yazpei/projets/compagnon_immo/MLE/Compagnon_immo/data' # YASMINE
    #folder_path = '' # KETSIA

    # Lire les fichiers exportés pour les logger
    output_folder = Path(folder_path)

    parquet_file = output_folder / "df_sales_clean_polars.parquet"
    csv_file = output_folder / "df_sales_clean_polars.csv"

    if parquet_file.exists():
        mlflow.log_artifact(str(parquet_file))
    if csv_file.exists():
        mlflow.log_artifact(str(csv_file))

    try:
        df = pd.read_csv(csv_file, sep=";", nrows=100000)
        mlflow.log_metric("rows_exported", df.shape[0])
        mlflow.log_metric("cols_exported", df.shape[1])
        mlflow.log_metric("missing_dates", df["date"].isna().sum())
    except Exception as e:
        print(f"Erreur lors de la lecture de {csv_file} : {e}")
