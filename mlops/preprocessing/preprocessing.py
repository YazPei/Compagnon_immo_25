import os
import click
import pandas as pd
import polars as pl
import mlflow
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# === Fonctions de nettoyage et utilitaires ===

def clean_exposition(x):
    if pd.isna(x):
        return np.nan
    x = x.lower()
    if 'nord' in x:
        return 'Nord'
    elif 'sud' in x:
        return 'Sud'
    elif 'est' in x:
        return 'Est'
    elif 'ouest' in x:
        return 'Ouest'
    return 'Autre'

def clean_classe(x):
    if pd.isna(x):
        return np.nan
    x = str(x).upper()
    return x if x in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else 'Autre'

def extract_principal(df):
    df['prix_total'] = df['prix_m2_vente'] * df['surface_reelle_bati']
    df['prix_moyen_piece'] = df['prix_total'] / df['nombre_pieces_principales']
    df['log_prix_m2'] = np.log1p(df['prix_m2_vente'])
    return df

def detect_outliers(df):
    return df[(df['prix_m2_vente'] > 100) & (df['prix_m2_vente'] < 20000)]

def calculate_bounds(series, iqr_factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - iqr_factor * iqr, q3 + iqr_factor * iqr

def compute_medians(df):
    return df.median(numeric_only=True)

def mark_outliers(df, column):
    lower, upper = calculate_bounds(df[column])
    return df[column].apply(lambda x: np.nan if x < lower or x > upper else x)

def clean_outliers(df, columns):
    for col in columns:
        df[col] = mark_outliers(df, col)
    return df

def get_numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def group_col(df):
    df['annee'] = pd.to_datetime(df['date_mutation'], errors='coerce').dt.year
    df['prix_m2_classe'] = pd.qcut(df['prix_m2_vente'], q=5, labels=False, duplicates='drop')
    return df


# === Pipeline principale CLI ===

@click.command()
@click.option('--input-path', type=click.Path(exists=True), prompt='📥 Fichier d’entrée fusionné')
@click.option('--output-path', type=click.Path(), prompt='📤 Fichier de sortie nettoyé')
def main(input_path, output_path):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("Preprocessing Données Immo")

    with mlflow.start_run(run_name="preprocessing_pipeline"):
        df = pl.read_csv(input_path, separator=";").to_pandas()

        df.drop_duplicates(inplace=True)
        df.dropna(subset=['prix_m2_vente', 'surface_reelle_bati', 'nombre_pieces_principales'], inplace=True)
        df = detect_outliers(df)
        df = df[df['surface_reelle_bati'] > 0]
        df = df[df['nombre_pieces_principales'] > 0]

        df['classe_energetique'] = df['classe_energetique'].apply(clean_classe)
        df['exposition_principale'] = df['exposition_principale'].apply(clean_exposition)

        df = extract_principal(df)
        df = group_col(df)

        numeric_cols = get_numeric_cols(df)
        df = clean_outliers(df, numeric_cols)

        # === Tracking MLflow ===
        mlflow.log_param("input_path", input_path)
        mlflow.log_metric("nb_rows", len(df))
        mlflow.log_metric("nb_cols", df.shape[1])
        mlflow.log_metric("nb_nas", df.isna().sum().sum())

        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        pl.from_pandas(df).write_csv(output_path, separator=";")
        mlflow.log_artifact(output_path)

        print(f"✅ Données sauvegardées dans : {output_path}")

if __name__ == '__main__':
    main()
