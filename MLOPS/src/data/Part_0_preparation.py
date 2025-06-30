import os
import time
from pathlib import Path
import click
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from numba import njit, prange


@click.command()
def main():
    ### PROMPTS CLI ###
    folder_path1 = click.prompt('📂 Chemin vers les données ventes', type=click.Path(exists=True))
    folder_path2 = click.prompt('📂 Chemin vers les données DVF', type=click.Path(exists=True))
    output_filepath = click.prompt('📁 Dossier de sortie', type=click.Path())

    ### Chargement des données ###
    input_file1 = os.path.join(folder_path1, 'merged_sales_data.csv')
    input_file2 = os.path.join(folder_path2, 'DVF_donnees_macroeco.csv')

    chunks1 = pd.read_csv(input_file1, sep=';', chunksize=100_000, on_bad_lines='skip', low_memory=False)
    df_sales_clean = pd.concat(chunks1)

    chunks2 = pd.read_csv(input_file2, sep=',', chunksize=100_000, on_bad_lines='skip', low_memory=False)
    df_dvf = pd.concat(chunks2)

    # Nettoyage types
    df_sales_clean = df_sales_clean.astype({col: str for col in df_sales_clean.select_dtypes(include='object').columns})
    df_sales_clean['INSEE_COM'] = df_sales_clean['INSEE_COM'].astype(str).str.zfill(5).str.strip()
    df_dvf['INSEE_COM'] = df_dvf['Code INSEE de la commune'].astype(str).str.strip()

    # Filtres IPS
    df_dvf = df_dvf[(df_dvf['Rentrée scolaire'] == '2023-2024') & (df_dvf['Secteur'] == 'public')]
    df_dvf['IPS'] = df_dvf['IPS'].astype(str).replace('NS', np.nan).str.replace(',', '.').astype(float)

    def extract_level(name):
        name = name.upper()
        if 'PRIMAIRE' in name:
            return 'primaire'
        if 'ELEMENTAIRE' in name:
            return 'elementaire'
        return np.nan

    df_dvf['level'] = df_dvf['Etablissement'].apply(extract_level)
    df_levels = df_dvf.dropna(subset=['level']).groupby(['INSEE_COM', 'level'])['IPS'].mean().reset_index()
    df_ips = df_levels.pivot(index='INSEE_COM', columns='level', values='IPS').rename(columns={
        'primaire': 'IPS_primaire',
        'elementaire': 'IPS_elementaire'
    }).reset_index()

    # Merge indicateurs économiques
    df_dvf_small = df_dvf[['avg_purchase_price_m2', 'avg_rent_price_m2', 'rental_yield_pct', 'INSEE_COM']].drop_duplicates('INSEE_COM')
    df_ips_small = df_ips[['INSEE_COM', 'IPS_primaire']].drop_duplicates('INSEE_COM')

    df_sales_clean['avg_purchase_price_m2'] = df_sales_clean['INSEE_COM'].map(dict(zip(df_dvf_small['INSEE_COM'], df_dvf_small['avg_purchase_price_m2'])))
    df_sales_clean['avg_rent_price_m2'] = df_sales_clean['INSEE_COM'].map(dict(zip(df_dvf_small['INSEE_COM'], df_dvf_small['avg_rent_price_m2'])))
    df_sales_clean['rental_yield_pct'] = df_sales_clean['INSEE_COM'].map(dict(zip(df_dvf_small['INSEE_COM'], df_dvf_small['rental_yield_pct'])))
    df_sales_clean['IPS_primaire'] = df_sales_clean['INSEE_COM'].map(dict(zip(df_ips_small['INSEE_COM'], df_ips_small['IPS_primaire'])))

    ### Imputation IPS manquants ###
    communes = df_sales_clean.groupby('INSEE_COM').agg({
        'mapCoordonneesLatitude': 'mean',
        'mapCoordonneesLongitude': 'mean',
        'IPS_primaire': 'mean'
    }).reset_index()

    complete = communes[communes['IPS_primaire'].notna()].copy()
    missing = communes[communes['IPS_primaire'].isna()].copy()

    tree = BallTree(np.radians(complete[['mapCoordonneesLatitude','mapCoordonneesLongitude']]), metric='haversine')
    dist, idx = tree.query(np.radians(missing[['mapCoordonneesLatitude','mapCoordonneesLongitude']]), k=1)
    missing['IPS_primaire'] = complete.iloc[idx.flatten()]['IPS_primaire'].values

    communes_filled = pd.concat([complete, missing]).drop_duplicates('INSEE_COM')
    df_sales_clean['IPS_primaire'] = df_sales_clean['INSEE_COM'].map(dict(zip(communes_filled['INSEE_COM'], communes_filled['IPS_primaire'])))

    ### Export ###
    df_pl = pl.from_pandas(df_sales_clean)
    Path(output_filepath).mkdir(parents=True, exist_ok=True)

    df_pl.write_csv(Path(output_filepath) / "df_sales_clean_polars.csv", separator=";")
    print("✅ Export CSV terminé :", Path(output_filepath) / "df_sales_clean_polars.csv")

if __name__ == '__main__':
    main()

