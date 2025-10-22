#!/usr/bin/env python


import logging
import math
# Standard library imports
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import mlflow
import polars as pl

warnings.filterwarnings("ignore")

# Géospatial imports
import geopandas as gpd
import matplotlib.pyplot as plt
# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Configuration de l'affichage pandas
pd.set_option('print.max_columns', None)  # Affiche toutes les colonnes
pd.set_option('print.width', 1000)       # Ajuste la largeur pour éviter les coupures
pd.set_option('print.colheader_justify', 'center')  # Centre les noms des colonnes

# Ignorer les warnings
warnings.filterwarnings('ignore')


class ClusteringService:
    """Service for clustering real estate data."""

    def __init__(self):
        pass

    def perform_clustering(self, input_path, output_path):
        """Perform clustering on the data."""
        # Placeholder for clustering logic
        print(f"Clustering data from {input_path} to {output_path}")
        return {"status": "success", "message": "Clustering completed"}

#############################################################################################

#############################################################################################
# 2. Configuration des chemins d'accès


@click.command()
@click.option('--input-path', type=click.Path(exists=True), prompt='📥 Fichier d’entrée nettoyé')
@click.option('--output-path', type=click.Path(), prompt='📤 Fichier clusterisé de sortie')
def main(input_path, output_path):
    print("📥 Lecture du fichier:", input_path)
    df = pl.read_csv(input_path, separator=";").to_pandas()

    # Définition des chemins d'accès aux données
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("Clustering Données Immo")
    with mlflow.start_run(run_name="clustering_macro_kpi"):
        # Définir le dossier contenant les fichiers d'entrée (par exemple, le dossier du fichier d'entrée)
        folder_path = os.path.dirname(input_path)
        # Chemins des fichiers
        train_file = os.path.join(folder_path, 'train_clean.csv')
        test_file = os.path.join(folder_path, 'test_clean.csv')
        geo_file = os.path.join(folder_path, 'data/contours-codes-postaux.geojson')

#############################################################################################


#############################################################################################
## 3. Chargement des données
    def load_data(file_path, chunksize=100000):
        """Charge les données à partir d'un fichier CSV avec gestion des encodages.

        Args:
            file_path (str): Chemin du fichier CSV
            chunksize (int): Taille des chunks pour le chargement

        Returns:
            DataFrame: Données chargées
        """
        encodings = ['utf-8', 'ISO-8859-1', 'latin1']

        for encoding in encodings:
            try:
                print(f"⏳ Tentative d'ouverture avec encodage : {encoding}")
                chunks = pd.read_csv(
                    file_path,
                    sep=";",
                    chunksize=chunksize,
                    index_col="date",
                    parse_dates=["date"],
                    on_bad_lines="skip",
                    low_memory=False,
                    encoding=encoding
                )
            # Process chunks
                data = pd.concat(chunk for chunk in chunks).sort_values(by="date")
                print(f"✅ Fichier lu avec succès avec encodage : {encoding}")
                return data
            except Exception as e:
                print(f"❌ Erreur avec encodage {encoding}: {e}")

        raise ValueError("Impossible de lire le fichier avec les encodages disponibles")

# Chargement des données d'entraînement
    print("Chargement des données d'entraînement...")
    train_cluster = load_data(train_file)
    train_cluster["Year"] = train_cluster.index.year
    train_cluster["Month"] = train_cluster.index.month
    train_cluster_ST = train_cluster[train_cluster["Year"]<'2024']
    test_cluster_ST = train_cluster[train_cluster["Year"] >= 2024]
# Chargement des données de test
    print("\nChargement des données de test...")
    test_cluster = load_data(test_file)


# #### Ajout de la variable code postal
    print("\nChargement des polygones de codes postaux...")



# === 2. CHARGEMENT DES POLYGONES DE CODES POSTAUX ===

    geo_cp_file = load_data(geo_file)



    pcodes = gpd.read_file(geo_cp_file)[['codePostal', 'geometry']]
    pcodes= pcodes.set_geometry('geometry')
    pcodes = pcodes.to_crs(epsg=4326)
    print("Polygones chargés :", pcodes.shape)

# Creation de l'index spatial pour accélérer la recherche
    _ = pcodes.sindex
    train_cluster_ST = train_cluster_ST.reset_index()

    train_cluster_ST['split'] = 'train' #(train for reg and ST)
    test_cluster_ST['split'] = 'train_test' #(train reg test ST)
    test_cluster['split'] = 'test' #(test for reg)

    # Combinaison des données pour le traitement
    df_cluster = pd.concat([train_cluster_ST, test_cluster_ST, test_cluster])

    # === 4. PRÉTRAITEMENT GEO ===
    df_base = df_cluster.copy()

    df_base = df_base.dropna(subset=['mapCoordonneesLatitude', 'mapCoordonneesLongitude'])
    df_base['lat'] = df_base['mapCoordonneesLatitude']#.round(3)
    df_base['lon'] = df_base['mapCoordonneesLongitude']#.round(3)
    df_base['orig_index'] = df_base.index

    # === 5. FONCTION DE TRAITEMENT SPATIAL D'UN CHUNK ===
    def process_chunk(chunk, pcodes):
        chunk = chunk.copy()
        chunk['geometry'] = gpd.points_from_xy(chunk['lon'], chunk['lat'])
        gdf = gpd.GeoDataFrame(chunk, geometry='geometry', crs='EPSG:4326')
        gdf =  gdf[gdf.is_valid]
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

    df_base['orig_index'] = df_base.index  # pour merge

    df_base = df_base.merge(df_joined[['orig_index', 'codePostal']], on="orig_index", how="left")
    df_base.drop(columns=['orig_index'], inplace=True)

    # === 8. VÉRIFICATION DU RÉSULTAT ===
    print(df_base[['mapCoordonneesLatitude', 'mapCoordonneesLongitude', 'codePostal', 'date']].head())
    print("Code postal manquant :", df_base['codePostal'].isna().sum())


    # #### Creation variable temps et cp




    df_base['date'] = pd.to_datetime(df_base['date'], errors='coerce')
    df_base = df_base.sort_values('date')

    # Définir la colonne 'date' comme index
    df_base = df_base.set_index('date')


# traiter le codePostal


    df_base["codePostal"] = df_base["codePostal"].astype(str).str.replace(r"\.0$", "", regex=True)


# Vérification des types de données
    print("\nTypes de données dans df_base :")
    print(df_base.dtypes)
# Vérification des colonnes datetime
    datetime_cols = df_base.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    print("\nColonnes de type datetime dans df_base :")
    if not datetime_cols:
        print("Aucune colonne de type datetime trouvée.")
    else:
        print(f"Nombre de colonnes datetime : {len(datetime_cols)}")
    # Afficher les valeurs uniques des colonnes datetime
    print("\nValeurs uniques des colonnes datetime :")
    for col in datetime_cols:
        print(f"Colonne datetime : {col}")
        print(df_base[col].unique())


# corriger les valeurs de la colonne 'codePostal'
    for code in df_base["codePostal"].unique():
        if len(str(code)) < 5:
            code = str(code).zfill(5)
        # Convert 'codePostal' to string
    df_base["codePostal"] = df_base["codePostal"].astype(str)
    print(df_base.head())





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




# On s'assure que les codes postaux sont bien au format 5 chiffres


    train_cluster = df_base[df_base['split']=='train']
    test_cluster = df_base[(df_base['split']== 'test') | (df_base['split']== 'train_test') ]

# On garde les codes postaux fréquents
    cp_counts = train_cluster["codePostal"].value_counts()
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
    train_cluster["zone_mixte"] = train_cluster.apply(
        lambda row: regroup_code(row, cp_frequents_str), axis=1
    )

# Pour test_cluster, on applique exactement la même logique sans recalculer les fréquences
    test_cluster["zone_mixte"] = test_cluster.apply(
        lambda row: regroup_code(row, cp_frequents_str), axis=1
    )


# ##### construction d'un jeu d'entrainement avec la variable 'Zone Mixte' et un lag -1



    train_cluster.sort_values(["zone_mixte", "date"], inplace=True)
    train_cluster["prix_lag_1m"] = (train_cluster.groupby("zone_mixte")["prix_m2_vente"].shift(1)
        )
    train_cluster["prix_roll_3m"] = (train_cluster.groupby("zone_mixte")["prix_m2_vente"]
            .rolling(3, closed="left")
            .mean()
            .reset_index(level=0, drop=True)
        )

# contruire un jeu train et test avec les zones mixtes par mois
    train_mensuel = (
        train_cluster.groupby(["Year", "Month", "zone_mixte"])
        .agg(
            prix_m2_vente =("prix_m2_vente", "mean"),
            volume_ventes=("prix_m2_vente", "count"),
        )
        .reset_index()
    )


# ### Création de variable propre à la segmentation géographique
# Ces variables vont évaluer la volatilité du prix, le taux de croissance, la moyenne des prix et la variabilité

# #### Calcul du taux de croissance annuel lissé #######

# L'objectif est de prendre en compte la tendance globale de l'évolution des prix par code postal, sur toute la période observée, en lissant les variations mois par mois.


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

    os.makedirs("mlflow_outputs", exist_ok=True)
    df_cluster_input.to_csv("mlflow_outputs/cluster_input.csv", index=False, sep=";")

    ###############################################
#    test_cluster.loc[mask_valid, ["codePostal_recons", "cluster", "cluster_label"]].to_csv("mlflow_outputs/test_clusters.csv", index=False, sep=";")

    # Log des artefacts
    mlflow.log_artifact("mlflow_outputs/cluster_input.csv")
    mlflow.log_artifact("mlflow_outputs/test_clusters.csv")
# # ##### Vérification de la qualité du taux de croissance annuel


#     from sklearn.linear_model import LinearRegression

# #  Visualisation de la tendance log-linéaire
# # On choisit un code postal pour visualiser la tendance
# print(train_mensuel["codePostal_recons"].unique())
# code_postal_exemple = "75019"  # à adapter selon tes données

# # Extraire les données correspondantes
# df_exemple = train_mensuel[
#     train_mensuel["codePostal_recons"] == code_postal_exemple
# ].dropna(subset=["log_prix", "t"])

# print(df_exemple.shape)
# print(df_exemple.head())
# # Fit de la régression
# X = df_exemple[["t"]]
# y = df_exemple["log_prix"]
# model = LinearRegression().fit(X, y)
# y_pred = model.predict(X)

# # Plot
# plt.figure(figsize=(10, 5))
# plt.scatter(df_exemple["t"], y, label="log(prix réel)", color="blue")
# plt.plot(df_exemple["t"], y_pred, label="régression linéaire", color="red", linewidth=2)
# plt.title(f"Tendance log-linéaire des prix pour le code postal {code_postal_exemple}")
# plt.xlabel("Temps (années depuis la première observation)")
# plt.ylabel("Log du prix au m²")
# plt.legend()
# plt.grid(True)
# plt.show()


# ## Clustering avec KMeans

# ### Recherche du nombre optimal de clusters


    from sklearn.cluster import KMeans
    from sklearn.model_selection import \
        train_test_split  # pour un clustering applicable aux 2 modèles
    from sklearn.preprocessing import StandardScaler

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
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    inertias = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        inertias.append(km.fit(X_train_scaled).inertia_)
    # Méthode du coude
    plt.figure()
    plt.plot(range(2, 10), inertias, marker="o")
    plt.title("Coude k-means – Inertie intra-cluster")
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mlflow_outputs/elbow_plot.png")
    mlflow.log_artifact("mlflow_outputs/elbow_plot.png")
    plt.close()

# --- 4. Fit KMeans définitif (ici k=4) ---

# --- 4. Clustering final + log MLflow ---
with mlflow.start_run(run_name="clustering_macro_kpi"):

    # Log des paramètres
    mlflow.log_params({
        "algo": "KMeans",
        "n_clusters": 4,
        "random_state": 42,
        "scaling": "StandardScaler",
        "features": ",".join(features)
    })

    # Méthode du coude
    inertias = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train_scaled)
        inertias.append(km.inertia_)

    os.makedirs("mlflow_outputs", exist_ok=True)
    plt.figure()
    plt.plot(range(2, 10), inertias, marker='o')
    plt.title("Méthode du coude – Inertie intra-cluster")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mlflow_outputs/elbow_plot.png")
    mlflow.log_artifact("mlflow_outputs/elbow_plot.png")
    plt.close()

    # Fit final
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_train_scaled)

    # Affectation dans df_cluster_input
    df_cluster_input.loc[train_idx, "cluster"] = labels.astype(int)

    # Sauvegarde
    df_cluster_input.to_csv("mlflow_outputs/cluster_input.csv", index=False, sep=";")
    mlflow.log_artifact("mlflow_outputs/cluster_input.csv")

# ── 5. Prédiction sur les lignes complètes du test ──
mask_valid = ~test_cluster[features].isna().any(axis=1)
X_test_valid = test_cluster.loc[mask_valid, features]
X_test_scaled = scaler.transform(X_test_valid)
test_cluster.loc[mask_valid, "cluster"] = kmeans.predict(X_test_scaled)

# ── 6. Mapping des clusters vers des labels lisibles ──
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

df_cluster_input["cluster_label"] = df_cluster_input["cluster"].map(mapping)
test_cluster.loc[mask_valid, "cluster_label"] = test_cluster.loc[mask_valid, "cluster"].map(mapping)

# Sauvegarde des prédictions test pour audit
test_cluster.loc[mask_valid, ["codePostal_recons", "cluster", "cluster_label"]].to_csv(
    "mlflow_outputs/test_clusters.csv", index=False, sep=";"
)
mlflow.log_artifact("mlflow_outputs/test_clusters.csv")

# ── 4. Filtrage des lignes complètes et prédiction ──
# On ne clusterise que les lignes sans NaN
mask_valid = ~test_cluster[features].isna().any(axis=1)
X_test_valid = test_cluster.loc[mask_valid, features]
X_test_scaled = scaler.transform(X_test_valid)

test_cluster.loc[mask_valid, "cluster"] = kmeans.predict(X_test_scaled)

# ### fixation des clusters

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
    test_cluster.loc[mask_valid, "cluster_label"] = test_cluster.loc[mask_valid, "cluster"].map(mapping)

# ── 6. Résultat ──
    print(test_cluster.loc[mask_valid, ["codePostal_recons"] + features + ["cluster", "cluster_label"]].head())
    print(f"{mask_valid.sum()} lignes sur {len(test_cluster)} assignées à un cluster.")


# ### Visualisation


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




    import matplotlib.patches as mpatches

# ── 1. Fonction “string-only” pour regrouper les codes postaux ──


# ── 2. Calculer les centroïdes (lat/lon moyennes) par codePostal ──
    coord_cp = (
        train_cluster
        .dropna(subset=["mapCoordonneesLatitude", "mapCoordonneesLongitude"])
        .groupby("codePostal")[["mapCoordonneesLatitude","mapCoordonneesLongitude"]]
        .mean()
        .reset_index()
    )

# ── 3. Appliquer le regroupement et reconstruire codePostal_recons ──
    coord_cp["zone_mixte"]        = coord_cp["codePostal"].astype(str).apply(
        lambda cp: regroup_code(cp, cp_frequents_str)
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
# Variance plus forte → marché dispersé, peut-être entre anciens et nouveaux quartiers, zones périurbaines en mutation.
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




# Etape 4: export des données
    print(test_cluster.head())
    print(test_cluster["cluster"].value_counts())
    # Et en pourcentage du total
    print(test_cluster["cluster"].value_counts(normalize=True) * 100)

    test_clean = test_cluster.copy()
    train_clean = train_cluster.copy()




# ## Export des datasets


# Combinaison des données pour le traitement
    df_cluster_ST = pd.concat([train_cluster, test_cluster]).drop(columns='split')
    df_cluster = pd.concat([train_cluster, test_cluster])

# # Enregistrer le DataFrame final
# Serie temporelle pour Part-2_ST

    output_filepath = click.prompt('Enter the file path for the output data with the clusters', type=click.Path())
    os.makedirs(os.path.join(output_filepath, "data"), exist_ok=True)
    df_cluster_ST.to_csv(os.path.join(output_filepath, "data/df_sales_clean_ST.csv"), sep=";", index=True)
    mlflow.log_artifact(output_filepath)

# regression pour Part-2_R
# Enregistrer les dataset Train_clean et test_clean
    df_cluster['split'] = df_cluster['split'].replace('train_test','train') #(train for reg)

    df_cluster.to_csv(os.path.join(output_filepath, "data/df_cluster.csv"), sep=";", index=True)
    mlflow.log_artifact(output_filepath)
class ClusteringService:
    """Service for clustering real estate data."""

    def __init__(self):
        pass

    def perform_clustering(self, input_path, output_path):
        """Perform clustering on the data."""
        # Placeholder for clustering logic
        print(f"Clustering data from {input_path} to {output_path}")
        return {"status": "success", "message": "Clustering completed"}

if __name__ == '__main__':
    main()
