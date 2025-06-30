##########################################################################################
# IMPORT DES LIBRAIRIES ET PACKAGES
# Jupyter magic
%matplotlib inline

# Standard library imports
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import BallTree
from numba import njit, prange
##########################################################################################

##########################################################################################
### CPU ###
def travail_lourd(x):
    if x == 5:
        raise ValueError("Oups, erreur volontaire pour x=5")
    time.sleep(1)
    return x * x

inputs = list(range(12))
results = []

max_workers = min(4, os.cpu_count() or 1)

with ThreadPoolExecutor(max_workers=max_workers) as exe:
    futures = {exe.submit(travail_lourd, i): i for i in inputs}
    for fut in as_completed(futures):
        i = futures[fut]
        try:
            res = fut.result()
        except Exception as e:
            print(f"Tâche {i} a levé une exception : {e}")
            res = None
        results.append(res)

print("Résultats :", results)
### CPU ###
@njit(parallel=True)
def somme_racines(n):
    tmp = np.zeros(n)
    for i in prange(n):
        tmp[i] = np.sqrt(i)
    return np.sum(tmp)

##########################################################################################

# Import DATASET Data_sales
####################### DATA MACRO ECO - DVF  / DEMAND #############################
## paths
@click.command()
@click.argument('folder_path1', type=click.Path(exists=False), required=0)
@click.argument('folder_path2', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)


folder_path1 = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))

# Load the dataset df_sales_clean / OFFER

input_file1 = os.path.join(folder_path1, 'merged_sales_data.csv')


chunksize = 100000  # Number of rows per chunk
chunks = pd.read_csv(input_file1, sep=';', chunksize=chunksize, index_col=None, on_bad_lines='skip', low_memory=False)
# Process chunks
df_sales_clean = pd.concat(chunk for chunk in chunks)

## Rappel des colonnes restantes
print("Colonnes restantes dans le DataFrame :")
print(df_sales_clean.columns)
print(df_sales_clean.dtypes)
print("\nShape du Dataset après élimination des colonnes :", df_sales_clean.shape)
print(df_sales_clean['INSEE_COM'].info())

display(df_sales_clean.head())


##########################################################################################


##########################################################################################
# Nettoyage des types de données annonces avant conversion

object_cols = df_sales_clean.select_dtypes(include='object').columns
for col in object_cols:
    df_sales_clean[col] = df_sales_clean[col].astype(str)

##########################################################################################

##########################################################################################
# IMPORT DATASET DVF MACRO ECO
folder_path2 = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))

input_file2 = os.path.join(folder_path2, 'DVF_donnees_macroeco.csv')


chunksize = 100000  # Number of rows per chunk
chunks = pd.read_csv(input_file2, sep=',', chunksize=chunksize, index_col=None, on_bad_lines='skip', low_memory=False)
# Process chunks
df_dvf = pd.concat(chunk for chunk in chunks)

print(df_sales_clean['INSEE_COM'].dtype)
print(df_dvf['Code INSEE de la commune'].dtype)

# Clean format
df_sales_clean['INSEE_COM'] = df_sales_clean['INSEE_COM'].astype(str).str.pad(width=5, side='left', fillchar='0')
df_dvf['INSEE_COM'] = df_dvf['Code INSEE de la commune'].astype(str)

# Verification
print(df_sales_clean['INSEE_COM'].unique()[:30])
print(df_dvf['INSEE_COM'].unique()[:30])

df_sales_clean['INSEE_COM'] = df_sales_clean['INSEE_COM'].str.strip()
df_dvf['INSEE_COM'] = df_dvf['INSEE_COM'].str.strip()
##########################################################################################


##########################################################################################
# nettoyage des données IPS : 
# on ne garde que la rentrée 2023-2024, le secteur public, et une moyenne IPS par code commune sur primaire et elementaire
# Pour le moment nous ne conserverons que les données primaires
print(df_dvf.columns.tolist())

df_dvf = df_dvf[df_dvf['Rentrée scolaire'] == '2023-2024']
df_dvf = df_dvf[df_dvf['Secteur'] == 'public']

df_schools = df_dvf.drop(columns = ['avg_purchase_price_m2', 'avg_rent_price_m2', 'rental_yield_pct', 'num_ligne'])
df_schools['IPS'] = df_schools['IPS'].astype(str).replace('NS', np.nan)
df_schools['IPS'] = (df_schools['IPS'].astype(str).str.replace(',', '.').astype(float))

# creation d'une colonne "levele = "Primaire" |"elementaire"
def extract_level(name):
    name = name.upper()
    if 'PRIMAIRE' in name:
        return 'primaire'
    if 'ELEMENTAIRE' in name:
        return 'elementaire'
    return np.nan

df_schools['level'] = df_schools['Etablissement'].apply(extract_level)

# Filtrer les lignes valides et grouper pour faire la moyenne par commune et par niveau
df_levels = (df_schools.dropna(subset=['level']).groupby(['INSEE_COM', 'level'])['IPS'].mean().reset_index())

# Pivot pour obtenir une ligne par code_com, deux colonnes distinctes
df_ips_par_commune = (df_levels.pivot(index='INSEE_COM', columns='level', values='IPS').rename(columns={
        'primaire': 'IPS_primaire',
        'elementaire': 'IPS_elementaire'
    }).reset_index())

# Affichage
print(df_ips_par_commune.head())

# merge
df_ips_small = df_ips_par_commune[['INSEE_COM', 'IPS_primaire']].drop_duplicates('INSEE_COM')
df_dvf_small = df_dvf[['avg_purchase_price_m2', 'avg_rent_price_m2', 'rental_yield_pct', 'INSEE_COM']].drop_duplicates('INSEE_COM')




# Dict de lookup pour chaque colonne
purchase_map = dict(zip(df_dvf_small['INSEE_COM'], df_dvf_small['avg_purchase_price_m2']))
rent_map     = dict(zip(df_dvf_small['INSEE_COM'], df_dvf_small['avg_rent_price_m2']))
yield_map    = dict(zip(df_dvf_small['INSEE_COM'], df_dvf_small['rental_yield_pct']))
ips_map = dict(zip(df_ips_small['INSEE_COM'], df_ips_small['IPS_primaire']))


# Mapping dans df_sales_clean (pas de copy de tout le DF)
df_sales_clean['avg_purchase_price_m2'] = df_sales_clean['INSEE_COM'].map(purchase_map)
df_sales_clean['avg_rent_price_m2']     = df_sales_clean['INSEE_COM'].map(rent_map)
df_sales_clean['rental_yield_pct']      = df_sales_clean['INSEE_COM'].map(yield_map)
df_sales_clean['IPS_primaire'] = df_sales_clean['INSEE_COM'].map(ips_map)


# display(df_sales_clean.isna().sum())
# display(df_sales_clean[df_sales_clean['avg_rent_price_m2'].isna()][['INSEE_COM', 'avg_purchase_price_m2', 'dpeL', 'prix_m2_vente']].head(20))
# #display(df_sales_clean[df_sales_clean['INSEE_COM']==]['INSEE_COM', 'avg_purchase_price_m2', 'dpeL', 'prix_m2_vente'].head(20))


##########################################################################################


##########################################################################################
# CHECKS
display(df_sales_clean.columns)

##########################################################################################

##########################################################################################
### Imputation des nan par les données geo les plus proches
# 1. Moyenne lat/lon et IPS_primaire par commune
communes = (
    df_sales_clean
    .groupby('INSEE_COM')
    .agg({
      'mapCoordonneesLatitude':'mean',
      'mapCoordonneesLongitude':'mean',
      'IPS_primaire':'mean'  # NaN restent NaN
    })
    .reset_index()
)

# 2. Séparer communes complètes vs manquantes
complete = communes[communes['IPS_primaire'].notna()].copy()
missing  = communes[communes['IPS_primaire'].isna()].copy()

# 3. NearestNeighbors sur complete (en radians)

coords = np.radians(complete[['mapCoordonneesLatitude','mapCoordonneesLongitude']])
tree = BallTree(coords, metric='haversine')

# 4. Pour chaque commune manquante, chercher la plus proche
coords_m = np.radians(missing[['mapCoordonneesLatitude','mapCoordonneesLongitude']])
dist, idx = tree.query(coords_m, k=1)
missing['IPS_primaire'] = complete.iloc[idx.flatten()]['IPS_primaire'].values

# Concat et merge back sur chaque annonce
communes_filled = pd.concat([complete, missing], ignore_index=True)[
    ['INSEE_COM', 'IPS_primaire']
].drop_duplicates('INSEE_COM')

# dict pour map  
ips_map = dict(zip(
    communes_filled['INSEE_COM'],
    communes_filled['IPS_primaire']
))

df_sales_clean['IPS_primaire_imputed'] = df_sales_clean['INSEE_COM'].map(ips_map)


df_sales_clean['IPS_primaire'] = df_sales_clean['IPS_primaire'].fillna(
    df_sales_clean['IPS_primaire_imputed']
)
df_sales_clean.drop(columns=['IPS_primaire_imputed'], inplace=True)

##########################################################################################

##########################################################################################
display(df_sales_clean.head())

output_filepath = click.prompt('Enter the file path for the output prepared data', type=click.Path())


## Export des données
# Conversion en DataFrame Polars
df_pl = pl.from_pandas(df_sales_clean)

# Export en Parquet
parquet_path = Path(output_filepath)/ "df_sales_clean_polars.parquet"
print(f"Export Parquet Polars : {parquet_path}")

# Export en CSV
csv_path = Path(output_filepath) / "df_sales_clean_polars.csv")
df_pl.write_csv(csv_path, separator=";")
print(f"Export CSV Polars : {csv_path}")

print("Exports Parquet et CSV Polars terminés.")
##########################################################################################
