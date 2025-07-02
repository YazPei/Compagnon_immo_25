#!/usr/bin/env python
# coding: utf-8

# # Série Temporelle

# ## Preprocessing

# ### Import du dataset et split temporel

# #### Import du dataset avec comme index date

# In[6]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from numba import njit, prange
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
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
@njit(parallel=True)
def somme_racines(n):
    tmp = np.zeros(n)
    for i in prange(n):
        tmp[i] = np.sqrt(i)
    return np.sum(tmp)


# In[8]:


import numpy as np
from numba import njit, prange

@njit(parallel=True)
def somme_racines(n):
    acc = 0.0
    for i in prange(n):
        acc += np.sqrt(i)
    return acc

# utilise tous les threads automatiquement
print(somme_racines(100_000_000))


# In[9]:


## paths
# folder_path_M = '/Users/maximehenon/Documents/GitHub/MAR25_BDS_Compagnon_Immo/'
folder_path_Y = "C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON"
# folder_path_C = '../data/processed/Sales'
# folder_path_L= '/Users/loick.d/Documents/Datascientest/Github immo/MAR25_BDS_Compagnon_Immo/'


# Load the dataset
# output_file = os.path.join(folder_path_M, 'df_sales_clean_ST.csv')
output_file = os.path.join(folder_path_Y, "df_sales_clean_ST.csv")
# output_file = os.path.join(folder_path_C, 'df_sales_clean_ST.csv')
# output_file = os.path.join(folder_path_L, 'df_sales_clean_ST.csv')

from tqdm import tqdm
import time

# Exemple d'une boucle avec une barre de progression
for i in tqdm(range(100)):
    time.sleep(0.1)  # Simuler un traitement long
    
chunksize = 100000  # Number of rows per chunk
chunks = pd.read_csv(
    output_file,
    sep=";",
    chunksize=chunksize,
    index_col="date",
    parse_dates=["date"],
    on_bad_lines="skip",
    low_memory=False,
)
# Process chunks
df_sales_clean_ST = pd.concat(chunk for chunk in chunks).sort_values(by="date")
print(df_sales_clean_ST.index.unique().sort_values().to_series().dt.day.unique())

print(df_sales_clean_ST.groupby("date")["INSEE_COM"].nunique().head(12))

display(df_sales_clean_ST.head())



# #### Ajout de la variable code postal

# In[ ]:


import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

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


# #### Agrégation mensuelle 

# In[12]:


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


# #### Split Train et Test

# In[13]:


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


# #### Analyse des tendances

# ##### Agrégation des données par mois

# In[14]:


train_clean["departement"] = train_clean["codePostal"].astype(str).str[:2]
test_clean["departement"] = test_clean["codePostal"].astype(str).str[:2]

train_copy = train_clean.copy()

train_mensuel = (
    train_copy.groupby(["Year", "Month", "departement", "codePostal"])
    .agg(prix_m2_vente=("prix_m2_vente", "mean"))
    .reset_index()
)




# ##### Formattage des données temporelles

# In[ ]:


# Pour le mensuel
train_mensuel["date"] = pd.to_datetime(
    train_mensuel["Year"].astype(str) + "-" + train_mensuel["Month"].astype(str) + "-01"
)


# ##### Visualisation

# In[15]:


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


# Pour mieux expliquer l'évolution de la Target, nous ajoutons les taux immobilier à notre set de donnée

# ## Enrichissement du dataset

# Pour mieux adresser le problème, nous allons procéder à la segmentation des departements afin d'adresser les prix par segment géographique
# Par exemple sur les biens immobiliers comme Paris, nous allons l'enrichir par des données de taux d'emprunt immobilier, taux de chomage, ...

# ### Extraction des indicateurs pour clustering

# #### Ajustement de la granularité pour le clustering
# 

# ##### Constat initial
# 
# Nous avons commencé par réaliser un clustering par code postal, en utilisant des indicateurs agrégés par `codePostal` (prix moyen, écart-type, taux de croissance annuel moyen, etc.).  
# Cependant, au fil de l'analyse, nous avons constaté que **de nombreux codes postaux disposaient de très peu de données**, parfois **moins de 5 ventes**.
# 
# Cela posait plusieurs problèmes :
# - Les statistiques calculées (moyenne, écart-type, TCAM) étaient **peu fiables**.
# - Ces points faiblement renseignés pouvaient **brouiller le clustering** global.
# 
# ##### Seuil critique observé
# 
# Nous avons observé que :
# - Certains `codePostal` n'avaient **qu'une seule entrée**.
# - Le seuil de **10 observations** est un minimum généralement admis pour calculer des agrégations fiables.
# 
# ##### Solution mise en œuvre : **agrégation hybride**
# 
# Pour conserver à la fois **la précision locale** quand elle est disponible, et **la stabilité statistique** ailleurs, nous avons adopté une stratégie hybride :
# 
# - Si un `codePostal` contient **au moins 10 observations**, il est **conservé tel quel**.
# - Sinon, il est **regroupé au niveau du département** (`codePostal[:2]`).
# 
# Nous avons donc créé une nouvelle colonne appelée `zone_mixte`, qui contient :
# - soit le code postal complet (`75001`, `13008`, etc.)
# - soit le code départemental (`30`, `32`, `97`, etc.)
# 
# ##### Objectif
# 
# Cette approche permet de :
# - **Préserver la finesse géographique** dans les zones bien renseignées,
# - **Limiter le bruit** dans les zones sous-représentées,
# - Améliorer la **qualité du clustering** sans perdre d'information utile.
# 
# 

# #### Création de la variable hybride 'Zone Mixte' - Departement & Code Postal

# In[16]:


# On s'assure que les codes postaux sont bien au format 5 chiffres

train_clean["date"] = pd.to_datetime(train_clean["date"])
test_clean ["date"] = pd.to_datetime(test_clean ["date"])

# On garde les codes postaux fréquents
cp_counts = train_clean["codePostal"].value_counts()
cp_frequents_str = set(cp_counts[cp_counts >= 10].index)


# Fonction hybride
def regroup_code(row, frequents_set):
    cp = row["codePostal"]
    if cp in frequents_set:
        return cp  # code postal détaillé
    elif cp.startswith("97"):
        return cp[:3]  # DROM-COM
    elif cp.isdigit() and len(cp) == 5:
        return cp[:2]  # département
    else:
        return "inconnu"


#  Application sur train et test
train_clean["zone_mixte"] = train_clean.apply(
    lambda row: regroup_code(row, cp_frequents_str), axis=1
)

# Pour test_clean, on applique exactement la même logique sans recalculer les fréquences
test_clean["zone_mixte"] = test_clean.apply(
    lambda row: regroup_code(row, cp_frequents_str), axis=1
)


# ##### construction d'un jeu d'entrainement avec la variable 'Zone Mixte' et un lag -1

# In[17]:


for df in (train_clean, test_clean):
    df.sort_values(["zone_mixte", "date"], inplace=True)
    df["prix_lag_1m"] = (
        df.groupby("zone_mixte")["prix_m2_vente"].shift(1)
    )
    df["prix_roll_3m"] = (
        df.groupby("zone_mixte")["prix_m2_vente"]
          .rolling(3, closed="left")
          .mean()
          .reset_index(level=0, drop=True)
    )

# contruire un jeu train et test avec les zones mixtes par mois
train_mensuel = (
    train_clean.groupby(["Year", "Month", "zone_mixte"])
    .agg(
        prix_m2_vente =("prix_m2_vente", "mean"),
        volume_ventes=("prix_m2_vente", "count"), 
    )
    .reset_index()
)


# ### Création de variable propre à la segmentation géographique
# Ces variables vont évaluer la volatilité du prix, le taux de croissance, la moyenne des prix et la variabilité

# #### Calcul du taux de croissance annuel lissé
# L'objectif est de prendre en compte la tendance globale de l'évolution des prix par code postal,
# sur toute la période observée, en lissant les variations mois par mois.
# 

# In[18]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Chargement / check initial ---
# train_mensuel doit exister et contenir au minimum :
# ['zone_mixte', 'Year', 'Month', 'prix_m2_vente']
assert all(col in train_mensuel.columns 
           for col in ["zone_mixte", "Year", "Month", "prix_m2_vente"]),  "Il manque des colonnes dans train_mensuel."

# ---  Reconstruction du code postal propre ---
def get_code_postal_final(zone):
    s = str(zone)
    if s.isdigit() and len(s) == 5:
        return s
    if s.isdigit() and len(s) == 2:
        return s + "000"
    if s.startswith("97") and len(s) == 3:
        return s + "00"
    return "inconnu"

train_mensuel = train_mensuel.copy()
train_mensuel["codePostal_recons"] = (
    train_mensuel["zone_mixte"].apply(get_code_postal_final)
)

# 2. Date, ordinal et temps t (en mois depuis le début)
train_mensuel["date"] = pd.to_datetime(
    train_mensuel["Year"].astype(str)
    + "-" +
    train_mensuel["Month"].astype(str).str.zfill(2)
    + "-01"
)
train_mensuel["ym_ordinal"] = train_mensuel["Year"] * 12 + train_mensuel["Month"]
train_mensuel = train_mensuel.sort_values(["codePostal_recons","date"])
train_mensuel["t"] = (
    train_mensuel
    .groupby("codePostal_recons")["ym_ordinal"]
    .transform(lambda x: x - x.min())
)

# 3. Lags et moyennes mobiles (rolling)
train_mensuel["prix_lag_1m"] = (
    train_mensuel
    .groupby("codePostal_recons")["prix_m2_vente"]
    .shift(1)
)
train_mensuel["prix_roll_3m"] = (
    train_mensuel
    .groupby("codePostal_recons")["prix_m2_vente"]
    .rolling(3, closed="left")
    .mean()
    .reset_index(level=0, drop=True)
)

# 4. Log-prix et TCAM
train_mensuel["log_prix"] = np.log(train_mensuel["prix_m2_vente"])

def compute_tcam(df):
    if len(df) < 2 or df["log_prix"].isna().any():
        return np.nan
    X = df[["t"]].values.reshape(-1,1)
    y = df["log_prix"].values
    coef = LinearRegression().fit(X, y).coef_[0]
    return (np.exp(coef) - 1) * 100 * 12

tcam_df = (
    train_mensuel
    .groupby("codePostal_recons")
    .apply(compute_tcam)
    .reset_index(name="tc_am_reg")
)

# 5. Assemblage final des features
train_mensuel = (
    train_mensuel
    .merge(tcam_df, on="codePostal_recons", how="left")
    .rename(columns={"prix_m2_vente": "prix_m2_mean"})
    # Ne drop que les lignes où tes features indispensables sont manquantes
    .dropna(subset=["prix_lag_1m", "prix_roll_3m", "tc_am_reg"])
    .reset_index(drop=True)
)

# On remet à jour le log et t au cas où tu en auras besoin
train_mensuel["log_prix"]   = np.log(train_mensuel["prix_m2_mean"])
train_mensuel["t"]          = (
    train_mensuel
    .groupby("codePostal_recons")["ym_ordinal"]
    .transform(lambda x: x - x.min())
)

# Voilà ton DataFrame propre, prêt pour clustering ou modélisation
print(train_mensuel.head())


# #### calcul des autres feature et integration du Taux de croissance annuel lissé

# In[19]:


df_cluster_input = (
    train_mensuel
    .groupby("codePostal_recons")
    .agg(
        # on agrège les moyennes mensuelles calculées plus tôt
        prix_m2_mean = ("prix_m2_mean", "mean"),
        prix_m2_std  = ("prix_m2_mean", "std"),
        prix_m2_max  = ("prix_m2_mean", "max"),
        prix_m2_min  = ("prix_m2_mean", "min"),
        # on agrège aussi tes lags & rolling
        avg_lag_1m   = ("prix_lag_1m",   "mean"),
        avg_roll_3m  = ("prix_roll_3m",  "mean"),
         )
    .assign(
        prix_m2_cv = lambda df: df["prix_m2_std"] / df["prix_m2_mean"]
    )
    .reset_index()
    # fusionne ensuite ton TCAM déjà calculé
    .merge(tcam_df, on="codePostal_recons", how="left")
)

print(df_cluster_input.head())


# ##### Vérification de la qualité du taux de croissance annuel

# In[20]:


from sklearn.linear_model import LinearRegression

#  Visualisation de la tendance log-linéaire
# On choisit un code postal pour visualiser la tendance
print(train_mensuel["codePostal_recons"].unique())
code_postal_exemple = "75019"  # à adapter selon tes données

# Extraire les données correspondantes
df_exemple = train_mensuel[
    train_mensuel["codePostal_recons"] == code_postal_exemple
].dropna(subset=["log_prix", "t"])

print(df_exemple.shape)
print(df_exemple.head())
# Fit de la régression
X = df_exemple[["t"]]
y = df_exemple["log_prix"]
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(df_exemple["t"], y, label="log(prix réel)", color="blue")
plt.plot(df_exemple["t"], y_pred, label="régression linéaire", color="red", linewidth=2)
plt.title(f"Tendance log-linéaire des prix pour le code postal {code_postal_exemple}")
plt.xlabel("Temps (années depuis la première observation)")
plt.ylabel("Log du prix au m²")
plt.legend()
plt.grid(True)
plt.show()


# ## Clustering avec KMeans

# ### Recherche du nombre optimal de clusters

# In[21]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # pour un clustering applicable aux 2 modèles

# --- 0. Liste des features de clustering ---
features = [
    "prix_m2_std",
    "prix_m2_max",
    "prix_m2_min",
    "tc_am_reg",
    "prix_m2_cv",
    "avg_roll_3m",
    "avg_lag_1m",
]

# --- 1. Préparer X_train & conserver l'index ---
X = (
    df_cluster_input[features]
    .replace([np.inf, -np.inf], np.nan)
)
X_train = X.dropna()
X_train, X_test = train_test_split(X_train, test_size=0.2, random_state=42)
train_idx = X_train.index

# --- 2. Standardisation ---
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# --- 3. Méthode du coude pour k de 2 à 9 ---
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    inertias.append(km.fit(X_train_scaled).inertia_)

plt.figure()
plt.plot(range(2, 10), inertias, marker="o")
plt.title("Coude k-means – Inertie intra-cluster")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie")
plt.grid(True)
plt.show()

# --- 4. Fit KMeans définitif (ici k=4) ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_train_scaled)

# On injecte ces labels DANS df_cluster_input
df_cluster_input.loc[train_idx, "cluster"] = labels.astype(int)



# ### Création du jeu de test avec les variables de train

# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── 1. Nettoyage des anciennes colonnes ──
to_drop = features + ["cluster", "cluster_label"]
test_clean = test_clean.drop(columns=to_drop, errors="ignore")

# ── 2. Recréation de zone_mixte et codePostal_recons ──
cp_counts    = train_clean["codePostal"].value_counts()
cp_frequents = set(cp_counts[cp_counts >= 10].index.astype(str))

test_clean["zone_mixte"] = test_clean.apply(
    lambda row: regroup_code(row, cp_frequents),
    axis=1
)
test_clean["codePostal_recons"] = test_clean["zone_mixte"].apply(get_code_postal_final)
test_clean.drop(columns=["zone_mixte"], inplace=True)

# ── 3. Fusion des features agrégées ──
test_clean = test_clean.merge(
    df_cluster_input[["codePostal_recons"] + features],
    on="codePostal_recons",
    how="left"
)

# Vérification que toutes les features sont présentes
missing = set(features) - set(test_clean.columns)
if missing:
    raise ValueError(f"Il manque ces colonnes dans test_clean avant clustering : {missing}")

# ── 4. Filtrage des lignes complètes et prédiction ──
# On ne clusterise que les lignes sans NaN
mask_valid = ~test_clean[features].isna().any(axis=1)
X_test_valid   = test_clean.loc[mask_valid, features]
X_test_scaled  = scaler.transform(X_test_valid)

test_clean.loc[mask_valid, "cluster"] = kmeans.predict(X_test_scaled)



# ### fixation des clusters

# In[23]:


# ── 5. Mapping vers un label lisible ──
cluster_order = (
    df_cluster_input
    .groupby("cluster")["prix_m2_mean"]
    .mean()
    .sort_values()
    .index
    .tolist()
)
cluster_names = [
    "Zones rurales, petites villes stagnantes",
    "Centres urbains établis, zones résidentielles",
    "Banlieues, zones mixtes",
    "Zones tendues - secteurs spéculatifs",
]
mapping = dict(zip(cluster_order, cluster_names))

df_cluster_input['cluster_label']=df_cluster_input['cluster'].map(mapping)
test_clean.loc[mask_valid, "cluster_label"] = test_clean.loc[mask_valid, "cluster"].map(mapping)

# ── 6. Résultat ──
print(test_clean.loc[mask_valid, ["codePostal_recons"] + features + ["cluster", "cluster_label"]].head())
print(f"{mask_valid.sum()} lignes sur {len(test_clean)} assignées à un cluster.")


# ### Visualisation

# In[24]:


cluster_palette = {
    "Zones rurales, petites villes stagnantes":    "#1f77b4",
    "Banlieues, zones mixtes":                    "#ff7f0e",
    "Centres urbains établis, zones résidentielles":"#2ca02c",
    "Zones tendues - secteurs spéculatifs":        "#d62728",
}

# visualisation
sns.pairplot(
    df_cluster_input,
    vars=features,
    hue="cluster_label",
    hue_order=list(cluster_palette.keys()),
    palette=cluster_palette,
    corner=True            # pour n’afficher que la moitié inférieure et gagner en lisibilité
)
plt.suptitle("Distribution des indicateurs par cluster (train)", y=1.02)
plt.show()



# | Cluster |  Couleur  | Niveau de prix |    Volatilité   |    Croissance (tc\_am\_reg)   | Interprétation économique                                       |
# | :-----: | :-------: | :------------: | :-------------: | :---------------------------: | :-------------------------------------------------------------- |
# |    0    |  🔵 Bleu  |   **Faible**   | **Très faible** | **Faible / parfois négative** | **Zones rurales / petites villes stagnantes**                   |
# |    1    | 🟠 Orange |  **Moyen-bas** |   **Modérée**   |          **Modérée**          | **Périphéries et banlieues**                   |
# |    2    |  🟢 Vert  | **Moyen-haut** |   **Modérée**   |      **Modérée à bonne**      | **Centres urbains établis, marchés résidentiels stables**       |
# |    3    |  🔴 Rouge | **Très élevé** |    **Élevée**   |           **Forte**           | **Zones tendues / spéculatives (luxe, hypercentre, littoral…)** |
# 

# ### Visualisation sur une map

# In[25]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import matplotlib.patches as mpatches

# ── 0. Préparer la liste des codes postaux fréquents ──
cp_counts       = train_clean["codePostal"].value_counts()
cp_frequents_str = set(cp_counts[cp_counts >= 10].index.astype(str))

# ── 1. Fonction “string-only” pour regrouper les codes postaux ──
def regroup_code_str(cp: str, freq_set: set) -> str:
    if cp in freq_set:
        return cp
    if cp.startswith("97"):
        return cp[:3]
    if cp.isdigit() and len(cp) == 5:
        return cp[:2]
    return "inconnu"

# ── 2. Calculer les centroïdes (lat/lon moyennes) par codePostal ──
coord_cp = (
    train_clean
    .dropna(subset=["mapCoordonneesLatitude", "mapCoordonneesLongitude"])
    .groupby("codePostal")[["mapCoordonneesLatitude","mapCoordonneesLongitude"]]
    .mean()
    .reset_index()
)

# ── 3. Appliquer le regroupement et reconstruire codePostal_recons ──
coord_cp["zone_mixte"]        = coord_cp["codePostal"].astype(str).apply(
    lambda cp: regroup_code_str(cp, cp_frequents_str)
)
coord_cp["codePostal_recons"] = coord_cp["zone_mixte"].apply(get_code_postal_final)

# ── 4. Fusionner avec votre df_cluster_input (qui porte cluster & cluster_label) ──
geo_df = pd.merge(
    df_cluster_input.reset_index(),  # attention: index doit devenir col. réindexez sinon
    coord_cp[["codePostal_recons","mapCoordonneesLatitude","mapCoordonneesLongitude"]],
    on="codePostal_recons",
    how="left"
).dropna(subset=["mapCoordonneesLatitude","mapCoordonneesLongitude"])

# ── 5. Transformer en GeoDataFrame ──
geometry = [
    Point(xy) for xy in zip(
        geo_df["mapCoordonneesLongitude"],
        geo_df["mapCoordonneesLatitude"]
    )
]
geo_df = gpd.GeoDataFrame(geo_df, geometry=geometry, crs="EPSG:4326")

# Optionnel : ne garder que la métropole
geo_df = geo_df[~geo_df["codePostal_recons"].str.startswith(("97","98"))]

# ── 6. Choisir une palette de couleurs sur les labels ──


# ── 7. Tracer la carte en boucle pour une légende propre ──
fig, ax = plt.subplots(figsize=(10,12))
for lbl, color in cluster_palette.items():
    subset = geo_df[geo_df["cluster_label"] == lbl]
    subset.plot(
        ax=ax,
        color=color,
        markersize=25,
        alpha=0.7,
        label=lbl
    )
ax.legend(title="Type de zone", loc="lower left", fontsize=10, title_fontsize=12)
ax.set_title("Clusters immobiliers en France métropolitaine", fontsize=14)
ax.axis("off")
plt.show()

# ── 8. Boxplots explicatifs par cluster ──
features_box = ["prix_m2_mean","prix_m2_std","tc_am_reg","prix_m2_cv","avg_roll_3m","avg_lag_1m"]
sns.set_theme(style="whitegrid", palette="pastel")

order = list(cluster_palette.keys())

for feat in features_box:
    plt.figure(figsize=(10, 6))
    
    ax = sns.boxplot(
        y="cluster_label",           # on bascule en horizontal
        x=feat,
        data=geo_df,
        order=order,
        palette=cluster_palette,
        notch=True,                  # crans
        showfliers=False,            # pas les outliers extrêmes
        width=0.6
    )
    
    # Superposer les moyennes
    means = geo_df.groupby("cluster_label")[feat].mean().reindex(order)
    for i, m in enumerate(means):
        ax.plot(m, i, marker="D", color="black", label="_nolegend_")
    
    ax.set_title(f"{feat} par cluster", fontsize=14)
    ax.set_xlabel(feat.replace("_", " "), fontsize=12)
    ax.set_ylabel("")  # on conserve seulement le label des clusters
    plt.tight_layout()
    plt.show()


# In[26]:


# Export des résultats pour l'integration dans le modèle LGBM
# 1. Sélectionner et dédupliquer le mapping
clusters_st = (
    df_cluster_input[["codePostal_recons", "cluster"]]
    .drop_duplicates(subset="codePostal_recons")
)
folder_path_Y = "C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON"
# 2. Sauvegarder dans un CSV pour réutilisation ultérieure
clusters_st.to_csv(os.path.join(folder_path_Y, "clusters_st.csv"), sep=";", index=True, encoding="utf-8")


print("Export clusters_st.csv généré avec",
      len(clusters_st), "entrées (zones).")


# Nous sommes bien sur les clusters suivants:
# 
# - **Cluster 3 (rouge) — zone de luxe / tendue** 
# Clairement séparé en haut à droite de presque tous les nuages de points.
# Prix très élevés (mean, max, min), dispersion (std) forte.
# TCAM (tc_am_reg) souvent positif.
# Très cohérent avec des zones chères, touristiques ou spéculatives.
# 
# 
# - **Cluster 0 (bleu) —  ville dense, mature** zones à prix modérément élevés mais stables
# Prix moyens comparables au cluster orange, voire légèrement supérieurs.
# Variance (écart-type) plus faible : le marché est plus homogène.
# TCAM souvent modéré → zones matures et stabilisées, comme des centres-villes établis ou zones résidentielles stables.
# 
# - **Cluster 1 (orange) — zones périurbaines ou dynamiques secondaires**:  ville mixte, périurbaine ou segmentée
# Prix moyens similaires au cluster bleu.
# Variance plus forte → marché plus dispersé, peut-être entre anciens et nouveaux quartiers, zones périurbaines en mutation.
# TCAM modérément positif → marchés dynamiques ou en rattrapage, mais moins structurés que le bleu.
# 
# 
# - **Cluster 2 (vert) — zones rurales / peu dynamiques**
# Prix très bas et très homogènes.
# Très peu de variance → marché stagnant.
# TCAM parfois négatif → zones en décroissance ou stagnation.
# 
# 

# ## Varifications et export des données

# In[27]:


# Etape 4: export des données
display(test_clean.head())
display(test_clean["cluster"].value_counts())
# Et en pourcentage du total
print(test_clean["cluster"].value_counts(normalize=True) * 100)


# ## Préparation Encodage des facteurs exogènes pour SARIMAX

# ### Création d'une liste de facteurs exogènes

# In[28]:


variables_exp = ["taux_rendement_n7", "loyer_m2_median_n7","y_geo", "x_geo", "z_geo", "dpeL", "nb_pieces", 'IPS_primaire','rental_yield_pct']


# Pour notre modélisation, nous allons choisir SARIMAX
# Pour cela nous aurons besoin de preprocesser et encoder nos facteurs exogènes

# ### Encodage des facteurs exogènes
# Nous allons prendre les Top 10 features issus de la feature selection (Part-2)
# 'taux_rendement_n7', 'loyer_m2_median_n7', 'y_geo', 'x_geo', 'z_geo', 'taux_rendement_n6', 'nb_pieces'
# 
# Nous allons également ajouter le taux d'emprunt 20 ans

# #### Facteurs exogènes : encodage et standardisation

# ##### Encodage de la variable géographique

# In[29]:


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


# ##### Encodage dpeL

# In[30]:


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


# ### Ajout de la variable cluster à train_data (donnée non agrégée)

# In[31]:


# 2. Positionnez df_cluster_input pour un mapping rapide
cluster_map = df_cluster_input.set_index("codePostal_recons")
train_clean["codePostal_recons"] = (train_clean["zone_mixte"].apply(get_code_postal_final))

train_clean['cluster'] = train_clean["codePostal_recons"].map(cluster_map['cluster'])

# 4. Vos autres features propres à train_clean restent :
#    "taux_rendement_n7", "loyer_m2_median_n7", "y_geo", "x_geo", "z_geo"

# Résultat : train_clean contient maintenant toutes les variables_exp
print(train_clean.head())



# 

# ##### Standardisation des facteurs exogènes

# In[32]:


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


# ### Creation d'un dataframe Monthly avec variables standardisés pour SARIMAX

# In[33]:


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


# #### Agrégation par mois et Création de la variable taux d'emprunt immobilier

# In[ ]:


# !pip install openpyxl


# In[34]:


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


# ## Export des datasets

# In[35]:


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

