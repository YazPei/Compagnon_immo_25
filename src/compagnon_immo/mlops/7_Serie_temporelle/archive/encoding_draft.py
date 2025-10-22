

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# 1 SPLIT 
# === 2. CHARGEMENT DES POLYGONES DE CODES POSTAUX ===
## Paths
# folder_path_M = '/Users/maximehenon/Documents/GitHub/MAR25_BDS_Compagnon_Immo/'
folder_path_Y = "C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON/"
#folder_path_C = '../data/geo/json'


geo_file_name = 'contours-codes-postaux.geojson'
input_file = os.path.join(folder_path_Y, geo_file_name)

pcodes = gpd.read_file(input_file)[['codePostal', 'geometry']]
pcodes= pcodes.set_geometry('geometry')
pcodes = pcodes.to_crs(epsg=4326) 
print("Polygones chargés :", pcodes.shape)

# Creation de l'index spatial pour accélérer la recherche
_ = pcodes.sindex
df_sales_clean_ST = df_sales_clean_ST.reset_index()

# === 4. PRÉTRAITEMENT GEO ===
df_base = df_sales_clean_ST.copy()

df_base = df_base.dropna(subset=['mapCoordonneesLatitude', 'mapCoordonneesLongitude'])
df_base['lat'] = df_base['mapCoordonneesLatitude']#.round(3)
df_base['lon'] = df_base['mapCoordonneesLongitude']#.round(3)
df_base['orig_index'] = df_base.index

# === 5. FONCTION DE TRAITEMENT SPATIAL D'UN CHUNK ===
def process_chunk(chunk, pcodes):
    chunk = chunk.copy()
    chunk['geometry'] = gpd.points_from_xy(chunk['lon'], chunk['lat'])
    gdf = gpd.GeoDataFrame(chunk, geometry='geometry', crs='EPSG:4326')
    gdf =  gdf.[gdf.is_valid]
    if gdf.crs != pcodes.crs:
        gdf = gdf.to_crs(pcodes.crs)
    _ = gdf.sindex 
    
    joined = gpd.sjoin(gdf, pcodes, how='left', predicate='within')
    return joined[['orig_index', 'codePostal']]  # retour minimal

# === 6. TRAITEMENT PAR CHUNKS POUR LIMITER LA MÉMOIRE ===
chunksize = 100_000
results = []

for i in range(0, len(df_base), chunksize):
    #print(f"Traitement du chunk {i} → {i+chunksize}")
    chunk = df_base.iloc[i:i+chunksize]
    result = process_chunk(chunk, pcodes)
    results.append(result)

# === 7. CONCATÉNATION DES RÉSULTATS ET MERGE FINAL ===
df_joined = pd.concat(results, ignore_index=True).drop_duplicates("orig_index")

df_sales_clean_ST['orig_index'] = df_sales_clean_ST.index  # pour merge

df_sales_clean_ST = df_sales_clean_ST.merge(df_joined[['orig_index', 'codePostal']], on="orig_index", how="left")
df_sales_clean_ST.drop(columns=['orig_index'], inplace=True)

# === 8. VÉRIFICATION DU RÉSULTAT ===
print(df_sales_clean_ST[['mapCoordonneesLatitude', 'mapCoordonneesLongitude', 'codePostal', 'date']].head())
print("Code postal manquant :", df_sales_clean_ST['codePostal'].isna().sum())


#### Agrégation mensuelle 
df_sales_clean_ST['date'] = pd.to_datetime(df_sales_clean_ST['date'], errors='coerce')
df_sales_clean_ST = df_sales_clean_ST.sort_values('date')

# Définir la colonne 'date' comme index
df_sales_clean_ST = df_sales_clean_ST.set_index('date')


# Creation des variable année et mois et traiter le codePostal

df_sales_clean_ST["Year"] = df_sales_clean_ST.index.year
df_sales_clean_ST["Month"] = df_sales_clean_ST.index.month

df_sales_clean_ST["codePostal"] = df_sales_clean_ST["codePostal"].astype(str).str.replace(r"\.0$", "", regex=True)

datetime_cols = df_sales_clean_ST.select_dtypes(include=["datetime64[ns]"]).columns

for col in datetime_cols:
    print(f"Colonne datetime : {col}")
    print(df_sales_clean_ST[col].unique())


# corriger les valeurs de la colonne 'codePostal'
for code in df_sales_clean_ST["codePostal"].unique():
    if len(str(code)) < 5:
        code = str(code).zfill(5)
    # Convert 'codePostal' to string
df_sales_clean_ST["codePostal"] = df_sales_clean_ST["codePostal"].astype(str)
display(df_sales_clean_ST.head())
#### Split Train et Test
# SPLIT

train_clean = df_sales_clean_ST[
    (df_sales_clean_ST["Year"] < 2024) & (df_sales_clean_ST["Year"] > 2019)
]
test_clean = df_sales_clean_ST[df_sales_clean_ST["Year"] >= 2024]

display(test_clean.head())
test_clean = test_clean.reset_index()
train_clean = train_clean.reset_index()
display(train_clean["codePostal"].head())
display(train_clean[train_clean["codePostal"] =="75019"].head())
#### Analyse des tendances
##### Agrégation des données par mois
train_clean["departement"] = train_clean["codePostal"].astype(str).str[:2]
test_clean["departement"] = test_clean["codePostal"].astype(str).str[:2]

train_copy = train_clean.copy()

train_mensuel = (
    train_copy.groupby(["Year", "Month", "departement", "codePostal"])
    .agg(prix_m2_vente=("prix_m2_vente", "mean"))
    .reset_index()
)


##### Formattage des données temporelles
# Pour le mensuel
train_mensuel["date"] = pd.to_datetime(
    train_mensuel["Year"].astype(str) + "-" + train_mensuel["Month"].astype(str) + "-01"
)
##### Visualisation
import plotly.express as px
import pandas as pd

Train_pour_graph = (
    train_clean.groupby(["Year", "Month"])
    .agg(prix_m2_vente=("prix_m2_vente", "mean"))
    .reset_index()
)

Train_pour_graph["date"] = pd.to_datetime(
    Train_pour_graph["Year"].astype(str)
    + "-"
    + Train_pour_graph["Month"].astype(str)
    + "-01"
)

###########################################
# Filtrage avec des dropdowns par departement

fig_mensuel_glob = px.line(
    Train_pour_graph,
    x="date",
    y="prix_m2_vente",
    title="Évolution mensuelle du prix moyen au m² ",
    labels={"date": "Date", "prix_m2_vente": "Prix moyen (€ / m²)"},
)

fig_mensuel_glob.update_traces(mode="lines+markers")
fig_mensuel_glob.update_layout(
    title_x=0.5,
    title_y=0.95,
    title_font_size=20,
    xaxis_title="Date",
    yaxis_title="Prix moyen (€ / m²)",
    hovermode="x unified",
)

fig_mensuel_glob.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
fig_mensuel_glob.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))
fig_mensuel_glob.show()

###########################################
# Filtrage avec des dropdowns par departement
Train_pour_graph_cp = (
    train_clean.groupby(["Year", "Month", "departement"])
    .agg(prix_m2_vente=("prix_m2_vente", "mean"))
    .reset_index()
)
Train_pour_graph_cp["date"] = pd.to_datetime(
    Train_pour_graph_cp["Year"].astype(str)
    + "-"
    + Train_pour_graph_cp["Month"].astype(str)
    + "-01"
)

fig_mensuel = px.line(
    Train_pour_graph_cp,
    x="date",
    y="prix_m2_vente",
    color="departement",
    title="Évolution mensuelle du prix moyen au m² par departement",
    labels={
        "date": "Date",
        "prix_m2_vente": "Prix moyen (€ / m²)",
        "departement": "departement",
    },
)

fig_mensuel.update_traces(mode="lines+markers")
fig_mensuel.update_layout(
    title_x=0.5,
    title_y=0.95,
    title_font_size=20,
    xaxis_title="Date",
    yaxis_title="Prix moyen (€ / m²)",
    legend_title_text="departement",
    hovermode="x unified",
)

fig_mensuel.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
fig_mensuel.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))

# Ajout de menus déroulants pour filtrer par departement et année
departement = train_mensuel["departement"].unique()

# Menu pour filtrer par departement
departement_buttons = [
    dict(
        label=str(cp),
        method="update",
        args=[
            {"visible": [cp == c for c in departement]},
            {"title": f"Évolution mensuelle pour le departement {cp}"},
        ],
    )
    for cp in departement
]

fig_mensuel.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=departement_buttons,
            direction="down",
            showactive=True,
            x=1.15,
            xanchor="left",
            y=1.1,
            yanchor="top",
            pad={"r": 10, "t": 10},
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )
    ]
)

fig_mensuel.show()


#2# Préparation Encodage des facteurs exogènes pour SARIMAX


### Création d'une liste de facteurs exogènes
variables_exp = ["taux_rendement_n7", "loyer_m2_median_n7","y_geo", "x_geo", "z_geo", "dpeL", "nb_pieces", 'IPS_primaire','rental_yield_pct']
### Encodage des facteurs exogènes
#Nous allons prendre les Top 10 features issus de la feature selection (Part-2)
#'taux_rendement_n7', 'loyer_m2_median_n7', 'y_geo', 'x_geo', 'z_geo', 'taux_rendement_n6', 'nb_pieces'

#Nous allons également ajouter le taux d'emprunt 20 ans, l'IPS
#### Facteurs exogènes : encodage et standardisation
##### Encodage de la variable géographique
# ENCODAGE DES VARIABLES GEOGRAPHIQUES
import numpy as np

lat_rad = np.radians(train_clean["mapCoordonneesLatitude"].values)
lon_rad = np.radians(train_clean["mapCoordonneesLongitude"].values)

# Projection sur la sphère unité :

### X_Train ###
train_clean["x_geo"] = np.cos(lat_rad) * np.cos(lon_rad)
train_clean["y_geo"] = np.cos(lat_rad) * np.sin(lon_rad)
train_clean["z_geo"] = np.sin(lat_rad)

### X_Test ###
lat_rad_test = np.radians(test_clean["mapCoordonneesLatitude"].values)
lon_rad_test = np.radians(test_clean["mapCoordonneesLongitude"].values)
test_clean["x_geo"] = np.cos(lat_rad_test) * np.cos(lon_rad_test)
test_clean["y_geo"] = np.cos(lat_rad_test) * np.sin(lon_rad_test)
test_clean["z_geo"] = np.sin(lat_rad_test)

# Les valeurs retournés sont comprises entre -1 et 1
# z est la latitude absolue (Nord /sud)
# x > 0 → vers l’Est (Greenwich → 90° E)
# x < 0 → vers l’Ouest (Greenwich → 90° O)
# y > 0 → moitié Nord de l’équateur (longitudes entre 0° et 180° E)
# y < 0 → moitié Sud (longitudes entre 0° et 180° O)

# suppression des colonnes Latitude et Longitude
train_clean = train_clean.drop(
    columns=["mapCoordonneesLongitude", "mapCoordonneesLatitude"]
)
test_clean = test_clean.drop(
    columns=["mapCoordonneesLongitude", "mapCoordonneesLatitude"]
)

# Verification
print(test_clean.head())
print(train_clean.head())
##### Encodage dpeL
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# creation d'une pipeline pour faire un imputer et un encodage
impute = SimpleImputer(strategy="most_frequent")
encode = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_clean["dpeL"] = train_clean["dpeL"].astype(str)
test_clean["dpeL"] = test_clean["dpeL"].astype(str)
# On crée une pipeline pour le prétraitement
pipeline = Pipeline(steps=[("imputer", impute), ("encoder", encode)])
# On applique la pipeline sur les colonnes catégorielles

train_clean["dpeL"] = pipeline.fit_transform(train_clean["dpeL"].values.reshape(-1, 1))
test_clean["dpeL"] = pipeline.transform(test_clean["dpeL"].values.reshape(-1, 1))

# Afficher les résultats
print("train_clean['dpeL'] après transformation :")
print(train_clean["dpeL"].unique())
print("test_clean['dpeL'] après transformation :")
print(test_clean["dpeL"].unique())

##### Standardisation des facteurs exogènes
# Standardisation des variables numériques
from sklearn.preprocessing import StandardScaler


# Créer une instance de StandardScaler
scaler = StandardScaler()

# Ajuster le scaler sur les données d'entraînement
train_clean[variables_exp] = scaler.fit_transform(train_clean[variables_exp])

# Appliquer la transformation sur les données de test
test_clean[variables_exp] = scaler.transform(test_clean[variables_exp])

# Vérification de la standardisation
print(train_clean[variables_exp].head())
print(test_clean[variables_exp].head())

### Creation d'un dataframe Monthly avec variables standardisés pour SARIMAX
# Regroupement mensuel par cluster – uniquement sur le train
variables_exp = [
    col for col in variables_exp if col not in ("cluster", "date")
]  # Regroupement mensuel par cluster (train uniquement)
# On regroupe par cluster et date
agg_cluster_monthly = (
    train_clean.groupby(["cluster", "date"], as_index=False)
    .agg({**{"prix_m2_vente": "mean"}, **{col: "mean" for col in variables_exp  }})
    .reset_index()
)

# Ajouter un indicateur split train/test pour plus tard (test sera prédit séparément)
agg_cluster_monthly["split"] = "train"

# Export sécurisé sans data leak
# agg_cluster_monthly.to_csv("agg_cluster_monthly.csv", index=False)



# Regroupement mensuel par cluster (test uniquement)
agg_cluster_monthly_test = test_clean.groupby(["cluster", "date"], as_index=False).agg(
    {"prix_m2_vente": "mean", **{col: "mean" for col in variables_exp}}
)
# Ajouter un indicateur split
agg_cluster_monthly_test["split"] = "test"

# Export pour inspection
# agg_cluster_monthly_test.to_csv("agg_cluster_monthly_test.csv", index=False)
#### Agrégation par mois et Création de la variable taux d'emprunt immobilier
# !pip install openpyxl
train_periodique_q12 = (
    agg_cluster_monthly[agg_cluster_monthly["split"] == "train"]
    .set_index("date")
    .drop(columns=["split"])
)
test_periodique_q12 = (
    agg_cluster_monthly_test[agg_cluster_monthly_test["split"] == "test"]
    .set_index("date")
    .drop(columns=["split"])
)


# display(train_periodique_q12.head(5))

##############################################################################
# Importer les données de taux d'intérêt
################################################################################
# Chemins d'accès aux fichiers

# chemin_taux_M = '/Users/maximehenon/Documents/GitHub/MAR25_BDS_Compagnon_Immo/data'
chemin_taux_Y = ("C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON/data")
# chemin_taux_C = '../data/banking'
# chemin_taux_L = '/Users/loick.d/Documents/Datascientest/Github immo/MAR25_BDS_Compagnon_Immo/data'

chemin_taux = os.path.join(chemin_taux_Y, "Taux immo.xlsx")
# chemin_taux = os.path.join(chemin_taux_C, 'Taux immo.xlsx')
# chemin_taux = os.path.join(chemin_taux_L, 'Taux immo.xlsx')
# chemin_taux = os.path.join(chemin_taux_M, 'Taux immo.xlsx')

# Importer les taux d'intérêt
import pandas as pd

taux = pd.read_excel(chemin_taux)
taux["date"] = pd.to_datetime(taux["date"], format="%Y-%m-%d")
taux = taux.set_index("date")
taux["taux"] = (
    taux["10 ans"].str.replace("%", "").str.replace(",", ".").str.strip().astype(float)
)
# display(taux.head(5))

# Fusionner les données de taux d'intérêt avec les données d'agrégation mensuelle
train_periodique_q12 = train_periodique_q12.merge(
    taux, left_index=True, right_index=True, how="left"
)
test_periodique_q12 = test_periodique_q12.merge(
    taux, left_index=True, right_index=True, how="left"
)

# Vérification de la fusion
# display(train_periodique_q12.head(5))


# Standardisation des taux d'intérêt
scal = StandardScaler()
train_periodique_q12["taux"] = scal.fit_transform(train_periodique_q12[["taux"]])
test_periodique_q12["taux"] = scal.transform(test_periodique_q12[["taux"]])
# Vérification de la standardisation
# print(train_periodique_q12.head())
# print(test_periodique_q12['taux'].head())
train_periodique_q12 = train_periodique_q12.reset_index()
test_periodique_q12 = test_periodique_q12.reset_index()

train_periodique_q12["prix_m2_vente"] = np.log(train_periodique_q12["prix_m2_vente"])
test_periodique_q12["prix_m2_vente"] = np.log(test_periodique_q12["prix_m2_vente"])


# Ne garder que les colonnes variables_exp
variables_exp = ["taux_rendement_n7", "loyer_m2_median_n7","y_geo", "x_geo", "z_geo", "dpeL", "nb_pieces", 'IPS_primaire','rental_yield_pct', 'taux']

train_periodique_q12 = train_periodique_q12[
    variables_exp + ["prix_m2_vente", "cluster", "date"]
]
test_periodique_q12 = test_periodique_q12[
    variables_exp + ["prix_m2_vente", "cluster", "date"]
]
# Vérification de la structure finale
print(train_periodique_q12.head())
print(test_periodique_q12.head())

## Export des datasets
# # Enregistrer le DataFrame final
train_periodique_q12.to_csv(
    os.path.join(folder_path_Y, "train_periodique_q12.csv"), sep=";", index=True
)
test_periodique_q12.to_csv(
    os.path.join(folder_path_Y, "test_periodique_q12.csv"), sep=";", index=True
)

# folder_path_M = ''
# folder_path_L = ''
# folder_path_C = '../data/processed/Sales'

# train_periodique_q12.to_csv(os.path.join(folder_path_C, 'train_periodique_q12.csv'), sep=';', index=True)
# test_periodique_q12.to_csv(os.path.join(folder_path_C, 'test_periodique_q12.csv'), sep=';', index=True)

# train_periodique_q12.to_csv(os.path.join(folder_path_M, 'train_periodique_q12.csv'), sep=';', index=True)
# test_periodique_q12.to_csv(os.path.join(folder_path_M, 'test_periodique_q12.csv'), sep=';', index=True)

# train_periodique_q12.to_csv(os.path.join(folder_path_L, 'train_periodique_q12.csv'), sep=';', index=True)
# test_periodique_q12.to_csv(os.path.join(folder_path_L, 'test_periodique_q12.csv'), sep=';', index=True)


# Enregistrer les dataset Train_clean et test_clean
train_clean.to_csv(os.path.join(folder_path_Y, "train_clean_ST.csv"), sep=";", index=True)
test_clean.to_csv(os.path.join(folder_path_Y, "test_clean_ST.csv"), sep=";", index=True)

# train_clean.to_csv(os.path.join(folder_path_C, "train_clean_ST.csv"), sep=";", index=True)
# test_clean.to_csv(os.path.join(folder_path_C, "test_clean_ST.csv"), sep=";", index=True)

# train_clean.to_csv(os.path.join(folder_path_M, 'train_clean.csv'), sep=';', index=True)
# test_clean.to_csv(os.path.join(folder_path_M, 'test_clean.csv'), sep=';', index=True)

# train_clean.to_csv(os.path.join(folder_path_L, 'train_clean.csv'), sep=';', index=True)
# test_clean.to_csv(os.path.join(folder_path_L, 'test_clean.csv'), sep=';', index=True)
