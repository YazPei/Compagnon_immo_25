# Generated from notebook: Part-2_ST_VF_final.ipynb
import mlflow
mlflow.set_experiment('companion_immo')


# ---- CODE CELL ----
# Jupyter magic
%matplotlib inline

# Standard library imports
import os
import time
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Geospatial imports
import geopandas as gpd
from shapely.geometry import Point

# Configuration de l'affichage pandas
pd.set_option('display.max_columns', None)  # Affiche toutes les colonnes
pd.set_option('display.width', 1000)       # Ajuste la largeur pour éviter les coupures
pd.set_option('display.colheader_justify', 'center')  # Centre les noms des colonnes

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Définition des chemins d'accès aux données
# Décommentez le chemin correspondant à votre environnement

# folder_path_M = '/Users/maximehenon/Documents/GitHub/MAR25_BDS_Compagnon_Immo/'
folder_path_Y = "C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON"
# folder_path_C = '../data/processed/Sales'
# folder_path_L = '/Users/loick.d/Documents/Datascientest/Github immo/MAR25_BDS_Compagnon_Immo/'
# folder_path_LW = 'C:/Users/User/Downloads/drive-download-20250508T155351Z-1-001'

# Utilisez cette variable pour définir votre chemin
folder_path = folder_path_Y  # Remplacez par votre variable de chemin

# Chemins des fichiers préparés par Part-1 bis
train_cluster_file = os.path.join(folder_path, 'train_cluster_prepared.csv')
train_mensuel_file = os.path.join(folder_path, 'train_mensuel_prepared.csv')
geo_file_name = 'contours-codes-postaux.geojson'
geo_file = os.path.join(folder_path, geo_file_name)

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
def load_prepared_data(file_path, index_col='date', parse_dates=True):
    """Charge les données préparées par Part-1 bis.
    
    Args:
        file_path (str): Chemin du fichier CSV
        index_col (str): Colonne à utiliser comme index
        parse_dates (bool): Si True, parse les dates
        
    Returns:
        DataFrame: Données chargées
    """
    try:
        print(f"⏳ Chargement des données depuis {file_path}")
        df = pd.read_csv(
            file_path,
            sep=";",
            index_col=index_col if index_col else None,
            parse_dates=True if parse_dates and index_col else None,
            low_memory=False
        )
        print(f"✅ Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        raise

# Chargement des données préparées
df_sales_clean_ST = load_prepared_data(train_cluster_file, index_col=None)

# Affichage des informations sur le dataset
print("\nAperçu des données:")
display(df_sales_clean_ST.head())

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Chargement des polygones de codes postaux
pcodes = gpd.read_file(geo_file)[['codePostal', 'geometry']]
print("Polygones chargés :", pcodes.shape)

# Prétraitement géo
df_base = df_sales_clean_ST.copy()
df_base = df_base.dropna(subset=['mapCoordonneesLatitude', 'mapCoordonneesLongitude'])
df_base['lat'] = df_base['mapCoordonneesLatitude']
df_base['lon'] = df_base['mapCoordonneesLongitude']
df_base['orig_index'] = df_base.index

# Fonction de traitement spatial d'un chunk
def process_chunk(chunk, pcodes):
    chunk = chunk.copy()
    chunk['geometry'] = gpd.points_from_xy(chunk['lon'], chunk['lat'])
    gdf = gpd.GeoDataFrame(chunk, geometry='geometry', crs='EPSG:4326')
    joined = gpd.sjoin(gdf, pcodes, how='left', predicate='within')
    return joined[['orig_index', 'codePostal']]  # retour minimal

# Traitement par chunks pour limiter la mémoire
chunksize = 100_000
results = []

for i in range(0, len(df_base), chunksize):
    chunk = df_base.iloc[i:i+chunksize]
    result = process_chunk(chunk, pcodes)
    results.append(result)

# Concaténation des résultats et merge final
df_joined = pd.concat(results, ignore_index=True).drop_duplicates("orig_index")
df_sales_clean_ST['orig_index'] = df_sales_clean_ST.index  # pour merge
df_sales_clean_ST = df_sales_clean_ST.merge(df_joined[['orig_index', 'codePostal']], on="orig_index", how="left")
df_sales_clean_ST.drop(columns=['orig_index'], inplace=True)

# Vérification du résultat
print(df_sales_clean_ST[['mapCoordonneesLatitude', 'mapCoordonneesLongitude', 'codePostal', 'date']].head())
print("Code postal manquant :", df_sales_clean_ST['codePostal'].isna().sum())

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Conversion de la colonne date en datetime
df_sales_clean_ST['date'] = pd.to_datetime(df_sales_clean_ST['date'], errors='coerce')
df_sales_clean_ST = df_sales_clean_ST.sort_values('date')

# Définir la colonne 'date' comme index
df_sales_clean_ST = df_sales_clean_ST.set_index('date')

# Création des variables année et mois et traitement du codePostal
df_sales_clean_ST["Year"] = df_sales_clean_ST.index.year
df_sales_clean_ST["Month"] = df_sales_clean_ST.index.month

# Conversion du code postal en string et nettoyage
df_sales_clean_ST["codePostal"] = df_sales_clean_ST["codePostal"].astype(str).str.replace(r"\.0$", "", regex=True)

# Vérification des colonnes datetime
datetime_cols = df_sales_clean_ST.select_dtypes(include=["datetime64[ns]"]).columns
for col in datetime_cols:
    print(f"Colonne datetime : {col}")
    print(df_sales_clean_ST[col].unique())

# Affichage des données formatées
display(df_sales_clean_ST.head())
display(df_sales_clean_ST["codePostal"].head())

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Agrégation nationale par mois
train_mensuel = (
    df_sales_clean_ST
    .groupby(["Year", "Month"])
    .agg(
        prix_m2_vente_mean=("prix_m2_vente", "mean"),
        prix_m2_vente_median=("prix_m2_vente", "median"),
        prix_m2_vente_std=("prix_m2_vente", "std"),
        prix_m2_vente_min=("prix_m2_vente", "min"),
        prix_m2_vente_max=("prix_m2_vente", "max"),
        prix_m2_vente_count=("prix_m2_vente", "count"),
        nb_transactions=("prix_m2_vente", "count")
    )
    .reset_index()
)

# Formattage des données temporelles
train_mensuel["date"] = pd.to_datetime(
    train_mensuel["Year"].astype(str) + "-" + train_mensuel["Month"].astype(str) + "-01"
)

# Affichage des données agrégées
display(train_mensuel.head())

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Extraction du département à partir du code postal
df_sales_clean_ST['departement'] = df_sales_clean_ST['codePostal'].str[:2]

# Correction pour les départements corses
df_sales_clean_ST.loc[df_sales_clean_ST['codePostal'].str.startswith('20'), 'departement'] = df_sales_clean_ST.loc[df_sales_clean_ST['codePostal'].str.startswith('20'), 'codePostal'].apply(
    lambda x: '2A' if x >= '20000' and x <= '20190' else '2B'
)

# Agrégation par département et par mois
dept_mensuel = (
    df_sales_clean_ST
    .groupby(["Year", "Month", "departement"])
    .agg(
        prix_m2_vente_mean=("prix_m2_vente", "mean"),
        prix_m2_vente_median=("prix_m2_vente", "median"),
        prix_m2_vente_std=("prix_m2_vente", "std"),
        nb_transactions=("prix_m2_vente", "count")
    )
    .reset_index()
)

# Formattage des dates
dept_mensuel["date"] = pd.to_datetime(
    dept_mensuel["Year"].astype(str) + "-" + dept_mensuel["Month"].astype(str) + "-01"
)

# Affichage des données agrégées par département
display(dept_mensuel.head())

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Filtrage des données pour Paris (75)
paris_data = df_sales_clean_ST.reset_index()
paris_data = paris_data[paris_data['codePostal'].str.startswith('75', na=False)]

# Extraction de l'arrondissement
paris_data['arrondissement'] = paris_data['codePostal'].str[2:].astype(int)

# Affichage des premières lignes pour Paris
display(paris_data.head())

# Agrégation par arrondissement et par mois
paris_mensuel = (
    paris_data
    .groupby(["Year", "Month", "arrondissement"])
    .agg(
        prix_m2_vente_mean=("prix_m2_vente", "mean"),
        nb_transactions=("prix_m2_vente", "count")
    )
    .reset_index()
)

# Formattage des dates
paris_mensuel["date"] = pd.to_datetime(
    paris_mensuel["Year"].astype(str) + "-" + paris_mensuel["Month"].astype(str) + "-01"
)

# Visualisation des prix par arrondissement
fig = px.line(
    paris_mensuel, 
    x="date", 
    y="prix_m2_vente_mean", 
    color="arrondissement",
    title="Évolution du prix moyen au m² par arrondissement de Paris",
    labels={"date": "Date", "prix_m2_vente_mean": "Prix moyen (€ / m²)", "arrondissement": "Arrondissement"}
)
fig.show()

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Visualisation de l'évolution du prix moyen au m²
fig = px.line(
    train_mensuel, 
    x="date", 
    y="prix_m2_vente_mean",
    title="Évolution du prix moyen au m² en France",
    labels={"date": "Date", "prix_m2_vente_mean": "Prix moyen (€ / m²)"}
)
fig.show()

# Visualisation du nombre de transactions
fig = px.line(
    train_mensuel, 
    x="date", 
    y="nb_transactions",
    title="Évolution du nombre de transactions immobilières en France",
    labels={"date": "Date", "nb_transactions": "Nombre de transactions"}
)
fig.show()

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Sélection des départements avec le plus de transactions
top_depts = dept_mensuel.groupby("departement")["nb_transactions"].sum().nlargest(10).index.tolist()
print(f"Top 10 départements par nombre de transactions : {top_depts}")

# Filtrage des données pour ces départements
top_dept_data = dept_mensuel[dept_mensuel["departement"].isin(top_depts)]

# Visualisation des prix par département
fig = px.line(
    top_dept_data, 
    x="date", 
    y="prix_m2_vente_mean", 
    color="departement",
    title="Évolution du prix moyen au m² par département",
    labels={"date": "Date", "prix_m2_vente_mean": "Prix moyen (€ / m²)", "departement": "Département"}
)
fig.show()

# Visualisation du nombre de transactions par département
fig = px.line(
    top_dept_data, 
    x="date", 
    y="nb_transactions", 
    color="departement",
    title="Évolution du nombre de transactions par département",
    labels={"date": "Date", "nb_transactions": "Nombre de transactions", "departement": "Département"}
)
fig.show()

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Agrégation par mois (tous les ans confondus)
monthly_agg = train_mensuel.groupby('Month').agg(
    prix_m2_vente_mean=('prix_m2_vente_mean', 'mean'),
    nb_transactions=('nb_transactions', 'mean')
).reset_index()

# Ajout des noms des mois
month_names = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
monthly_agg['month_name'] = monthly_agg['Month'].apply(lambda x: month_names[x-1])

# Visualisation de la saisonnalité des prix
fig = px.bar(
    monthly_agg, 
    x='month_name', 
    y='prix_m2_vente_mean',
    title="Prix moyen au m² par mois (saisonnalité)",
    labels={'month_name': 'Mois', 'prix_m2_vente_mean': 'Prix moyen (€ / m²)'}
)
fig.update_xaxes(categoryorder='array', categoryarray=month_names)
fig.show()

# Visualisation de la saisonnalité des transactions
fig = px.bar(
    monthly_agg, 
    x='month_name', 
    y='nb_transactions',
    title="Nombre moyen de transactions par mois (saisonnalité)",
    labels={'month_name': 'Mois', 'nb_transactions': 'Nombre moyen de transactions'}
)
fig.update_xaxes(categoryorder='array', categoryarray=month_names)
fig.show()

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Import des bibliothèques nécessaires
from statsmodels.tsa.seasonal import seasonal_decompose

# Préparation des données pour la décomposition
ts_data = train_mensuel.set_index('date')['prix_m2_vente_mean']

# Décomposition de la série temporelle
decomposition = seasonal_decompose(ts_data, model='additive', period=12)

# Création d'une figure avec 4 sous-graphiques
fig = make_subplots(rows=4, cols=1, subplot_titles=('Série originale', 'Tendance', 'Saisonnalité', 'Résidus'))

# Ajout des composantes à la figure
fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Original'), row=1, col=1)
fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, mode='lines', name='Tendance'), row=2, col=1)
fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, mode='lines', name='Saisonnalité'), row=3, col=1)
fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid, mode='lines', name='Résidus'), row=4, col=1)

# Mise en forme de la figure
fig.update_layout(height=900, title_text="Décomposition de la série temporelle des prix immobiliers")
fig.show()

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Import des bibliothèques nécessaires
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Préparation des données
ts_data = train_mensuel.set_index('date')['prix_m2_vente_mean']

# Split train/test
train_size = int(len(ts_data) * 0.8)
train_ts = ts_data[:train_size]
test_ts = ts_data[train_size:]

# Modèle ARIMA
model = ARIMA(train_ts, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Prévisions
forecast = model_fit.forecast(steps=len(test_ts))

# Évaluation
mse = mean_squared_error(test_ts, forecast)
mae = mean_absolute_error(test_ts, forecast)
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")

# Visualisation des prévisions
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_ts.index, y=train_ts.values, mode='lines', name='Données d\'entraînement'))
fig.add_trace(go.Scatter(x=test_ts.index, y=test_ts.values, mode='lines', name='Données de test'))
fig.add_trace(go.Scatter(x=test_ts.index, y=forecast, mode='lines', name='Prévisions ARIMA'))
fig.update_layout(title='Prévisions ARIMA des prix immobiliers', xaxis_title='Date', yaxis_title='Prix moyen (€ / m²)')
fig.show()

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Modèle SARIMA (ARIMA saisonnier)
sarima_model = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

# Prévisions SARIMA
sarima_forecast = sarima_fit.forecast(steps=len(test_ts))

# Évaluation SARIMA
sarima_mse = mean_squared_error(test_ts, sarima_forecast)
sarima_mae = mean_absolute_error(test_ts, sarima_forecast)
print(f"SARIMA MSE: {sarima_mse:.2f}")
print(f"SARIMA MAE: {sarima_mae:.2f}")

# Comparaison des performances
print(f"Amélioration MSE: {(mse - sarima_mse) / mse * 100:.2f}%")
print(f"Amélioration MAE: {(mae - sarima_mae) / mae * 100:.2f}%")

# Visualisation des prévisions SARIMA
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_ts.index, y=train_ts.values, mode='lines', name='Données d\'entraînement'))
fig.add_trace(go.Scatter(x=test_ts.index, y=test_ts.values, mode='lines', name='Données de test'))
fig.add_trace(go.Scatter(x=test_ts.index, y=sarima_forecast, mode='lines', name='Prévisions SARIMA'))
fig.update_layout(title='Prévisions SARIMA des prix immobiliers', xaxis_title='Date', yaxis_title='Prix moyen (€ / m²)')
fig.show()

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Entraînement du modèle sur toutes les données
full_model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
full_model_fit = full_model.fit(disp=False)

# Prévisions pour les 12 prochains mois
future_steps = 12
future_forecast = full_model_fit.forecast(steps=future_steps)

# Création des dates futures
last_date = ts_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')

# Visualisation des prévisions futures
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Données historiques'))
fig.add_trace(go.Scatter(x=future_dates, y=future_forecast, mode='lines', name='Prévisions futures', line=dict(dash='dash')))
fig.update_layout(
    title='Prévisions des prix immobiliers pour les 12 prochains mois',
    xaxis_title='Date',
    yaxis_title='Prix moyen (€ / m²)',
    shapes=[
        dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=last_date,
            y0=0,
            x1=future_dates[-1],
            y1=1,
            fillcolor="lightgray",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
        dict(
            x=last_date + (future_dates[-1] - last_date)/2,
            y=1.05,
            xref="x",
            yref="paper",
            text="Période de prévision",
            showarrow=False,
        )
    ]
)
fig.show()

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Vérification de la présence de la colonne cluster
if 'cluster' in df_sales_clean_ST.columns or 'cluster_label' in df_sales_clean_ST.columns:
    # Détermination du nom de la colonne cluster
    cluster_col = 'cluster' if 'cluster' in df_sales_clean_ST.columns else 'cluster_label'
    
    # Agrégation par cluster et par mois
    cluster_mensuel = (
        df_sales_clean_ST
        .groupby(["Year", "Month", cluster_col])
        .agg(
            prix_m2_vente_mean=("prix_m2_vente", "mean"),
            nb_transactions=("prix_m2_vente", "count")
        )
        .reset_index()
    )
    
    # Formattage des dates
    cluster_mensuel["date"] = pd.to_datetime(
        cluster_mensuel["Year"].astype(str) + "-" + cluster_mensuel["Month"].astype(str) + "-01"
    )
    
    # Visualisation des prix par cluster
    fig = px.line(
        cluster_mensuel, 
        x="date", 
        y="prix_m2_vente_mean", 
        color=cluster_col,
        title="Évolution du prix moyen au m² par cluster",
        labels={"date": "Date", "prix_m2_vente_mean": "Prix moyen (€ / m²)", cluster_col: "Cluster"}
    )
    fig.show()
    
    # Visualisation du nombre de transactions par cluster
    fig = px.line(
        cluster_mensuel, 
        x="date", 
        y="nb_transactions", 
        color=cluster_col,
        title="Évolution du nombre de transactions par cluster",
        labels={"date": "Date", "nb_transactions": "Nombre de transactions", cluster_col: "Cluster"}
    )
    fig.show()
else:
    print("Aucune colonne de cluster n'a été trouvée dans les données.")

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# Export des données mensuelles nationales
train_mensuel.to_csv(os.path.join(folder_path, 'train_mensuel_analysis.csv'), sep=';', index=False)
print(f"Données mensuelles nationales exportées vers {os.path.join(folder_path, 'train_mensuel_analysis.csv')}")

# Export des données mensuelles par département
dept_mensuel.to_csv(os.path.join(folder_path, 'dept_mensuel_analysis.csv'), sep=';', index=False)
print(f"Données mensuelles par département exportées vers {os.path.join(folder_path, 'dept_mensuel_analysis.csv')}")

# Export des prévisions
forecast_df = pd.DataFrame({
    'date': future_dates,
    'prix_m2_vente_forecast': future_forecast
})
forecast_df.to_csv(os.path.join(folder_path, 'prix_forecast.csv'), sep=';', index=False)
print(f"Prévisions exportées vers {os.path.join(folder_path, 'prix_forecast.csv')}")

with mlflow.start_run(run_name='Part-2_ST_VF_final'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)

