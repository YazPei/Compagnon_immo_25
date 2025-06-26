# Clustering - Préparation des données pour les analyses


## Introduction

#Ce notebook centralise le prétraitement et le clustering des données immobilières qui seront utilisés dans les notebooks suivants (régression et séries temporelles). L'objectif est d'éviter la duplication de code et d'assurer une cohérence dans les analyses.


#############################################################################################
## 1. Imports et configuration

# Jupyter magic
%matplotlib inline

# Standard library imports
import os
import re
import time
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Géospatial imports
import geopandas as gpd
from shapely.geometry import Point

# Configuration de l'affichage pandas
pd.set_option('display.max_columns', None)  # Affiche toutes les colonnes
pd.set_option('display.width', 1000)       # Ajuste la largeur pour éviter les coupures
pd.set_option('display.colheader_justify', 'center')  # Centre les noms des colonnes

# Ignorer les warnings
warnings.filterwarnings('ignore')

#############################################################################################

#############################################################################################
# 2. Configuration des chemins d'accès

# Définition des chemins d'accès aux données
# Décommentez le chemin correspondant à votre environnement

# folder_path_K = ''
# folder_path_Y = "C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON"

# Utilisez cette variable pour définir votre chemin
folder_path = folder_path_Y  # Remplacez par votre variable de chemin

# Chemins des fichiers
train_file = os.path.join(folder_path, 'train_clean.csv')
test_file = os.path.join(folder_path, 'test_clean.csv')
geo_file = os.path.join(folder_path, 'contours-codes-postaux.geojson')

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

# Chargement des données de test
print("\nChargement des données de test...")
test_cluster = load_data(test_file)

# Ajout d'une colonne pour identifier l'origine (train/test)
train_cluster['split'] = 'train'
test_cluster['split'] = 'test'

# Combinaison des données pour le traitement
train_cluster = pd.concat([train_cluster, test_cluster])

print("\nShape du Dataset:", train_cluster.shape)
display(train_cluster.head())

#############################################################################################################################

#############################################################################################################################
## 4. Enrichissement géospatial - Ajout des codes postaux
# Chargement des polygones de codes postaux
try:
    pcodes = gpd.read_file(geo_file)[['codePostal', 'geometry']]
    print("Polygones chargés :", pcodes.shape)
except Exception as e:
    print(f"Erreur lors du chargement du fichier géospatial: {e}")
    raise

# Préparation des données pour le traitement géospatial
train_cluster = train_cluster.reset_index(drop=False)  # Reset index pour éviter les problèmes de fusion
df_base = train_cluster.copy()
df_base = df_base.dropna(subset=['mapCoordonneesLatitude', 'mapCoordonneesLongitude'])
df_base['lat'] = df_base['mapCoordonneesLatitude']
df_base['lon'] = df_base['mapCoordonneesLongitude']
df_base['orig_index'] = df_base.index

# Fonction pour traiter un chunk de données
def process_chunk(chunk, pcodes):
    """Traite un chunk de données pour ajouter les codes postaux.
    
    Args:
        chunk (DataFrame): Chunk de données à traiter
        pcodes (GeoDataFrame): DataFrame contenant les polygones des codes postaux
        
    Returns:
        DataFrame: Résultat avec les codes postaux
    """
    chunk = chunk.copy()
    chunk['geometry'] = gpd.points_from_xy(chunk['lon'], chunk['lat'])
    gdf = gpd.GeoDataFrame(chunk, geometry='geometry', crs='EPSG:4326')
    joined = gpd.sjoin(gdf, pcodes, how='left', predicate='within')
    return joined[['orig_index', 'codePostal']]  # retour minimal

# Traitement par chunks pour limiter la consommation mémoire
chunksize = 100_000
results = []

for i in range(0, len(df_base), chunksize):
    chunk = df_base.iloc[i:i+chunksize]
    result = process_chunk(chunk, pcodes)
    results.append(result)

# Concaténation des résultats et fusion avec le DataFrame original
df_joined = pd.concat(results, ignore_index=True).drop_duplicates("orig_index")
train_cluster['orig_index'] = train_cluster.index  # pour merge
train_cluster = train_cluster.merge(df_joined[['orig_index', 'codePostal']], on="orig_index", how="left")
train_cluster.drop(columns=['orig_index'], inplace=True)

# Vérification du résultat
print(train_cluster[['mapCoordonneesLatitude', 'mapCoordonneesLongitude', 'codePostal', 'date']].head())
print("Code postal manquant :", train_cluster['codePostal'].isna().sum())

#################################################################################################################################


#################################################################################################################################	
## 5. Préparation des données pour l'analyse

#Cette section prépare les données en définissant la date comme index et en ajoutant des variables temporelles.
# Définir la colonne 'date' comme index
train_cluster = train_cluster.set_index('date')

# Création des variables année et mois
train_cluster["Year"] = train_cluster.index.year
train_cluster["Month"] = train_cluster.index.month

# Traitement du code postal
train_cluster["codePostal"] = train_cluster["codePostal"].astype(str).str.replace(r"\.0$", "", regex=True)

# Correction des codes postaux de longueur insuffisante
def format_code_postal(code):
    if pd.isna(code):
        return code
    code = str(code)
    if len(code) < 5:
        return code.zfill(5)
    return code

train_cluster["codePostal"] = train_cluster["codePostal"].apply(format_code_postal)

# Affichage des données préparées
display(train_cluster.head())
#################################################################################################################################	


#################################################################################################################################	
## 6. Création de la variable hybride 'Zone Mixte'

### Constat initial

#Nous avons constaté que de nombreux codes postaux disposent de très peu de données, parfois moins de 10 ventes. Cela pose plusieurs problèmes :
#- Les statistiques calculées (moyenne, écart-type, TCAM) sont peu fiables.
#- Ces points faiblement renseignés peuvent brouiller le clustering global.

### Solution mise en œuvre : agrégation hybride

#Pour conserver à la fois la précision locale quand elle est disponible, et la stabilité statistique ailleurs, nous adoptons une stratégie hybride :

#- Si un code postal contient au moins 10 observations, il est conservé tel quel.
#- Sinon, il est regroupé au niveau du département (les 2 premiers chiffres du code postal).

#Cette approche permet de :
#- Préserver la finesse géographique dans les zones bien renseignées
#- Limiter le bruit dans les zones sous-représentées
#- Améliorer la qualité du clustering sans perdre d'information utile

# Réinitialisation de l'index pour faciliter les manipulations
train_cluster_clean = train_cluster.reset_index()

# Ajout de la colonne département
train_cluster_clean["departement"] = train_cluster_clean["codePostal"].astype(str).str[:2]

# Identification des codes postaux fréquents (au moins 10 observations)
cp_counts = train_cluster_clean["codePostal"].value_counts()
cp_frequents_str = set(cp_counts[cp_counts >= 10].index)

# Fonction pour créer la zone mixte
def regroup_code(row, frequents_set):
    """Crée une zone mixte basée sur le code postal ou le département.
    
    Args:
        row (Series): Ligne du DataFrame
        frequents_set (set): Ensemble des codes postaux fréquents
        
    Returns:
        str: Code postal ou code département selon la fréquence
    """
    cp = row["codePostal"]
    if pd.isna(cp):
        return "inconnu"
    
    cp = str(cp)
    if cp in frequents_set:
        return cp  # code postal détaillé
    elif cp.startswith("97") and len(cp) >= 3:
        return cp[:3]  # DROM-COM
    elif cp.isdigit() and len(cp) == 5:
        return cp[:2]  # département
    else:
        return "inconnu"

# Application de la fonction pour créer la zone mixte
train_cluster_clean["zone_mixte"] = train_cluster_clean.apply(
    lambda row: regroup_code(row, cp_frequents_str), axis=1
)

# Affichage des résultats
print("Nombre de zones mixtes créées :", train_cluster_clean["zone_mixte"].nunique())
print("\nExemples de zones mixtes :")
display(train_cluster_clean[["codePostal", "departement", "zone_mixte"]].sample(10))

############################################################################################################################

############################################################################################################################
## 7. Agrégation des données par mois et zone mixte
# Agrégation mensuelle par zone mixte
train_mensuel = (
    train_cluster_clean.groupby(["Year", "Month", "zone_mixte"])
    .agg(
        prix_m2_mean=("prix_m2_vente", "mean"),
        volume_ventes=("prix_m2_vente", "count")
    )
    .reset_index()
)

# Reconstruction du code postal pour les analyses
def get_code_postal_final(zone):
    """Reconstruit un code postal à partir d'une zone mixte.
    
    Args:
        zone (str): Zone mixte (code postal ou département)
        
    Returns:
        str: Code postal reconstruit
    """
    s = str(zone)
    if s.isdigit() and len(s) == 5:
        return s
    if s.isdigit() and len(s) == 2:
        return s + "000"
    if s.startswith("97") and len(s) == 3:
        return s + "00"
    return "inconnu"

train_mensuel["codePostal_recons"] = train_mensuel["zone_mixte"].apply(get_code_postal_final)

# Création de la date complète
train_mensuel["date"] = pd.to_datetime(
    train_mensuel["Year"].astype(str) + "-" + 
    train_mensuel["Month"].astype(str).str.zfill(2) + "-01"
)

# Affichage des résultats
print("Données mensuelles agrégées :")
display(train_mensuel.head())
############################################################################################################################

############################################################################################################################
## 8. Création des variables pour le clustering

### 8.1 Calcul du taux de croissance annuel lissé (TCAM)

#L'objectif est de prendre en compte la tendance globale de l'évolution des prix par code postal, sur toute la période observée, en lissant les variations mois par mois.
# Création de variables temporelles pour l'analyse
train_mensuel["ym_ordinal"] = train_mensuel["Year"] * 12 + train_mensuel["Month"]
train_mensuel = train_mensuel.sort_values(["codePostal_recons", "date"])
train_mensuel["t"] = (
    train_mensuel
    .groupby("codePostal_recons")["ym_ordinal"]
    .transform(lambda x: x - x.min())
)

# Calcul des lags et moyennes mobiles
train_mensuel["prix_lag_1m"] = (
    train_mensuel
    .groupby("codePostal_recons")["prix_m2_mean"]
    .shift(1)
)
train_mensuel["prix_roll_3m"] = (
    train_mensuel
    .groupby("codePostal_recons")["prix_m2_mean"]
    .rolling(3, closed="left")
    .mean()
    .reset_index(level=0, drop=True)
)

# Calcul du logarithme des prix pour le TCAM
train_mensuel["log_prix"] = np.log(train_mensuel["prix_m2_mean"])

# Fonction pour calculer le TCAM par régression linéaire
def compute_tcam(df):
    """Calcule le taux de croissance annuel moyen par régression linéaire.
    
    Args:
        df (DataFrame): DataFrame contenant les colonnes t et log_prix
        
    Returns:
        float: TCAM en pourcentage annualisé
    """
    if len(df) < 2 or df["log_prix"].isna().any():
        return np.nan
    X = df[["t"]].values.reshape(-1, 1)
    y = df["log_prix"].values
    coef = LinearRegression().fit(X, y).coef_[0]
    return (np.exp(coef) - 1) * 100 * 12  # Annualisation

# Calcul du TCAM par code postal
tcam_df = (
    train_mensuel
    .groupby("codePostal_recons")
    .apply(compute_tcam)
    .reset_index(name="tc_am_reg")
)

# Fusion du TCAM avec les données mensuelles
train_mensuel = (
    train_mensuel
    .merge(tcam_df, on="codePostal_recons", how="left")
    # Ne drop que les lignes où les features indispensables sont manquantes
    .dropna(subset=["prix_lag_1m", "prix_roll_3m", "tc_am_reg"])
    .reset_index(drop=True)
)

# Affichage des résultats
print("Données avec TCAM calculé :")
display(train_mensuel.head())

## 8.2 Calcul des autres features pour le clustering
# Création du DataFrame pour le clustering
df_cluster_input = (
    train_mensuel
    .groupby("codePostal_recons")
    .agg(
        # Statistiques sur les prix
        prix_m2_mean=("prix_m2_mean", "mean"),
        prix_m2_std=("prix_m2_mean", "std"),
        prix_m2_max=("prix_m2_mean", "max"),
        prix_m2_min=("prix_m2_mean", "min"),
        # Statistiques sur les lags et rolling
        avg_lag_1m=("prix_lag_1m", "mean"),
        avg_roll_3m=("prix_roll_3m", "mean"),
    )
    .assign(
        # Coefficient de variation (mesure de dispersion relative)
        prix_m2_cv=lambda df: df["prix_m2_std"] / df["prix_m2_mean"]
    )
    .reset_index()
    # Fusion avec le TCAM déjà calculé
    .merge(tcam_df, on="codePostal_recons", how="left")
)

# Affichage des features pour le clustering
print("Features pour le clustering :")
display(df_cluster_input.head())
############################################################################################################################

############################################################################################################################
## 9. Visualisation des distributions des features
# Sélection des features numériques pour le clustering
features_for_clustering = [
    "prix_m2_mean", "prix_m2_std", "prix_m2_cv", "tc_am_reg"
]  ### A revoir Ketsia , ici il y a un probleme de lag!!!

# Visualisation des distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(features_for_clustering):
    sns.histplot(df_cluster_input[feature].dropna(), kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution de {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Fréquence")

plt.tight_layout()
plt.show()

# Matrice de corrélation
plt.figure(figsize=(10, 8))
corr_matrix = df_cluster_input[features_for_clustering].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matrice de corrélation des features")
plt.tight_layout()
plt.show()
############################################################################################################################

############################################################################################################################
## 10. Clustering KMeans
# Préparation des données pour le clustering
df_for_clustering = df_cluster_input.dropna(subset=features_for_clustering).copy()

# Standardisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_for_clustering[features_for_clustering])

# Détermination du nombre optimal de clusters avec la méthode du coude
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualisation de la méthode du coude
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Graphique de l'inertie
ax1.plot(k_range, inertia, 'o-')
ax1.set_xlabel('Nombre de clusters')
ax1.set_ylabel('Inertie')
ax1.set_title('Méthode du coude')
ax1.grid(True)

# Graphique du score de silhouette
ax2.plot(k_range, silhouette_scores, 'o-')
ax2.set_xlabel('Nombre de clusters')
ax2.set_ylabel('Score de silhouette')
ax2.set_title('Score de silhouette')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Choix du nombre optimal de clusters
optimal_k = 4  # À ajuster en fonction des résultats

# Application du clustering avec le nombre optimal de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_for_clustering['cluster'] = kmeans.fit_predict(X_scaled)

# Affichage des résultats
print(f"Clustering effectué avec {optimal_k} clusters")
print("\nRépartition des clusters :")
print(df_for_clustering['cluster'].value_counts())

# Caractéristiques des clusters
cluster_stats = df_for_clustering.groupby('cluster')[features_for_clustering].mean()
print("\nCaractéristiques moyennes des clusters :")
display(cluster_stats)
############################################################################################################################

############################################################################################################################
## 11. Visualisation des clusters
# Visualisation des clusters en fonction des features principales
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Graphique 1: Prix moyen vs TCAM
scatter1 = axes[0].scatter(
    df_for_clustering['prix_m2_mean'],
    df_for_clustering['tc_am_reg'],
    c=df_for_clustering['cluster'],
    cmap='viridis',
    alpha=0.7,
    s=50
)
axes[0].set_xlabel('Prix moyen au m²')
axes[0].set_ylabel('Taux de croissance annuel moyen (%)')
axes[0].set_title('Clusters par prix moyen et TCAM')
axes[0].grid(True, alpha=0.3)

# Graphique 2: Coefficient de variation vs Prix moyen
scatter2 = axes[1].scatter(
    df_for_clustering['prix_m2_mean'],
    df_for_clustering['prix_m2_cv'],
    c=df_for_clustering['cluster'],
    cmap='viridis',
    alpha=0.7,
    s=50
)
axes[1].set_xlabel('Prix moyen au m²')
axes[1].set_ylabel('Coefficient de variation des prix')
axes[1].set_title('Clusters par prix moyen et variabilité')
axes[1].grid(True, alpha=0.3)

# Légende commune
legend1 = axes[0].legend(*scatter1.legend_elements(), title="Clusters")
axes[0].add_artist(legend1)

plt.tight_layout()
plt.show()

# Visualisation des distributions par cluster
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(features_for_clustering):
    for cluster in range(optimal_k):
        sns.kdeplot(
            df_for_clustering[df_for_clustering['cluster'] == cluster][feature],
            ax=axes[i],
            label=f'Cluster {cluster}'
        )
    axes[i].set_title(f'Distribution de {feature} par cluster')
    axes[i].set_xlabel(feature)
    axes[i].legend()

plt.tight_layout()
plt.show()
############################################################################################################################

############################################################################################################################
## 12. Interprétation des clusters

#Analysons les caractéristiques de chaque cluster pour leur donner une interprétation métier.
# Interprétation des clusters
cluster_names = {
    0: "Marché ", ## Ketsia ici faudra revoir ! 
    1: "Marché ",
    2: "Marché",
    3: "Marché "
}

# Ajustez les noms en fonction des caractéristiques réelles de vos clusters
# Ces noms sont des exemples et doivent être adaptés à vos résultats

# Création d'un DataFrame avec les caractéristiques et les noms des clusters
cluster_profiles = cluster_stats.copy()
cluster_profiles['nom_cluster'] = cluster_profiles.index.map(cluster_names)
cluster_profiles = cluster_profiles.reset_index()

# Affichage des profils de clusters
print("Profils des clusters :")
display(cluster_profiles)

# Exemples de zones dans chaque cluster
print("\nExemples de zones par cluster :")
for cluster in range(optimal_k):
    examples = df_for_clustering[df_for_clustering['cluster'] == cluster]['codePostal_recons'].sample(min(5, sum(df_for_clustering['cluster'] == cluster))).tolist()
    print(f"Cluster {cluster} ({cluster_names[cluster]}): {examples}")
############################################################################################################################

############################################################################################################################
## 13. Fusion des clusters avec les données originales
# Création d'un dictionnaire de mapping code postal -> cluster
cp_to_cluster = dict(zip(df_for_clustering['codePostal_recons'], df_for_clustering['cluster']))

# Fonction pour attribuer un cluster à chaque code postal
def assign_cluster(cp):
    """Attribue un cluster à un code postal.
    
    Args:
        cp (str): Code postal
        
    Returns:
        int: Numéro de cluster ou -1 si non trouvé
    """
    if pd.isna(cp):
        return -1
    
    cp = str(cp)
    # Essai direct avec le code postal
    if cp in cp_to_cluster:
        return cp_to_cluster[cp]
    
    # Essai avec le code postal reconstruit (pour les départements)
    if len(cp) >= 2:
        dept_code = cp[:2] + "000"
        if dept_code in cp_to_cluster:
            return cp_to_cluster[dept_code]
    
    # Pour les DROM-COM
    if cp.startswith("97") and len(cp) >= 3:
        drom_code = cp[:3] + "00"
        if drom_code in cp_to_cluster:
            return cp_to_cluster[drom_code]
    
    return -1  # Cluster inconnu

# Application du clustering aux données originales
train_cluster_clean['cluster'] = train_cluster_clean['codePostal'].apply(assign_cluster)

# Ajout des noms de clusters
train_cluster_clean['nom_cluster'] = train_cluster_clean['cluster'].map(lambda x: cluster_names.get(x, "Non classé"))

# Vérification de la répartition
print("Répartition des clusters dans les données originales :")
print(train_cluster_clean['cluster'].value_counts(dropna=False))

# Affichage d'un échantillon
print("\nÉchantillon des données avec clusters :")
display(train_cluster_clean[['codePostal', 'departement', 'zone_mixte', 'cluster', 'nom_cluster', 'prix_m2_vente']].sample(10))
############################################################################################################################


############################################################################################################################
## 14. Agrégation mensuelle pour l'analyse de séries temporelles
# Agrégation mensuelle par département et code postal
train_mensuel_final = (
    train_cluster_clean.groupby(["Year", "Month", "departement", "codePostal", "cluster", "nom_cluster", "split"])
    .agg(
        prix_m2_vente=("prix_m2_vente", "mean"),
        nb_transactions=("prix_m2_vente", "count")
    )
    .reset_index()
)

# Création de la date complète
train_mensuel_final["date"] = pd.to_datetime(
    train_mensuel_final["Year"].astype(str) + "-" + 
    train_mensuel_final["Month"].astype(str).str.zfill(2) + "-01"
)

# Affichage des résultats
print("Données mensuelles finales :")
display(train_mensuel_final.head())
############################################################################################################################


############################################################################################################################
## 15. Export des données préparées
# Export des données avec clusters
train_cluster_clean.to_csv(os.path.join(folder_path, 'train_cluster_prepared.csv'), sep=';', index=False)
print(f"Données avec clusters exportées vers {os.path.join(folder_path, 'train_cluster_prepared.csv')}")

# Export des données mensuelles
train_mensuel_final.to_csv(os.path.join(folder_path, 'train_mensuel_prepared.csv'), sep=';', index=False)
print(f"Données mensuelles exportées vers {os.path.join(folder_path, 'train_mensuel_prepared.csv')}")

# Export des profils de clusters
cluster_profiles.to_csv(os.path.join(folder_path, 'cluster_profiles.csv'), sep=';', index=False)
print(f"Profils des clusters exportés vers {os.path.join(folder_path, 'cluster_profiles.csv')}")
############################################################################################################################


############################################################################################################################
## 16. Conclusion

#Dans ce notebook, nous avons réalisé les étapes suivantes :

#1. Chargement et préparation des données immobilières
#2. Enrichissement géospatial avec les codes postaux
#3. Création d'une variable hybride 'zone_mixte' pour équilibrer précision et fiabilité
#4. Calcul d'indicateurs avancés comme le taux de croissance annuel moyen (TCAM) # Ketsia, on doit revoir ici--> les KPI n'ont pas été respectés!!!
#5. Clustering des zones géographiques selon leurs caractéristiques de marché
#6. Interprétation et profilage des clusters identifiés
#7. Export des données préparées pour les analyses ultérieures

#Ces données enrichies et segmentées serviront de base pour les analyses de régression et de séries temporelles dans les notebooks suivants, permettant une compréhension plus fine des dynamiques du marché immobilier par segment.
############################################################################################################################

