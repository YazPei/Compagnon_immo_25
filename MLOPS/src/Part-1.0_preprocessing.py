#!/usr/bin/env python
# coding: utf-8

# # EXPLORATION

# In[1]:


# Jupyter magic
get_ipython().run_line_magic('matplotlib', 'inline')

# Standard library imports
import os
import re
import time
from numba import prange
import math
from numba import njit
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import train_test_split


# In[2]:


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


# ## Extrait et Shape du Dataset


@click.command()
@click.argument('folder_path', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)

## Paths

folder_path= click.prompt('Enter the file path for the input data', type=click.Path(exists=True))


## Load dataset

input_file = os.path.join(folder_path, 'df_sales_clean_polars.csv')


chunksize = 100000  # Number of rows per chunk
# chunks = pd.read_csv(input_file, sep=';',  chunksize=chunksize, index_col=None, on_bad_lines='skip', low_memory=False, encoding='ISO-8859-1')

def try_read_csv(path, sep=";", chunksize=100000):
    encodings_to_try = ['ISO-8859-1', 'latin1', 'utf-8']
    for encoding in encodings_to_try:
        try:
            print(f"⏳ Tentative d'ouverture avec encodage : {encoding}")
            chunks = pd.read_csv(
                path,
                sep=sep,
                chunksize=chunksize,
                index_col=None,
                on_bad_lines='skip',
                low_memory=False,
                encoding=encoding
            )
            df = pd.concat(chunk for chunk in chunks)
            print(f"✅ Fichier lu avec succès avec encodage : {encoding}")
            return df
        except UnicodeDecodeError as e:
            print(f"⚠️ Échec avec encodage {encoding} : {e}")
        except Exception as e:
            print(f"❌ Autre erreur : {e}")
    raise ValueError("Aucun encodage valide n'a permis d'ouvrir le fichier.")


# Utilisation
df_sales_clean = try_read_csv(input_file)
df_sales_clean['date'] = pd.to_datetime(df_sales_clean['date'], errors='coerce')


# Vérification
print("✅ Shape du DataFrame :", df_sales_clean.shape)
display(df_sales_clean.head(5))
# # Process chunks
# df_sales = pd.concat(chunk for chunk in chunks)

# Configurer Pandas pour afficher toutes les colonnes
pd.set_option('display.max_columns', None)  # Affiche toutes les colonnes
pd.set_option('display.width', 1000)       # Ajuste la largeur pour éviter les coupures
pd.set_option('display.colheader_justify', 'center')  # Centre les noms des colonnesµ

print("\n","Shape du Dataset",df_sales_clean.shape, "\n")


# In[4]:


print(df_sales_clean['date'].unique(), "\n")


# ## Gestion des Doublons

# In[5]:


print("Nombres de lignes en double", df_sales_clean.duplicated().sum())

df_sales_clean.drop_duplicates(inplace=True)

print("Nombres de lignes en double après suppression", df_sales_clean.duplicated().sum())
print("Shape du Dataset après élimination des doublons : ",df_sales_clean.shape)


# ## Gestion des NANs

# ### Proportions des NANs

# In[6]:


missing_data_percentage_sales = df_sales_clean.isna().sum()*100/len(df_sales_clean)

missing_value_percentage_sales = pd.DataFrame({'column_name': df_sales_clean.columns,
                                         'percent_missing': missing_data_percentage_sales,
                                         'dtypes':df_sales_clean.dtypes}
                                         ).sort_values(by='percent_missing', ascending=False)

# Resetting the index to start from 1 for better readability
# and to match the original DataFrame's index
missing_value_percentage_sales.index = range(1, len(missing_value_percentage_sales) + 1)

display(missing_value_percentage_sales)


# ### Visualisation des NANs

# In[7]:


plt.figure(figsize=(10, 14))

sns.barplot(
    y=missing_value_percentage_sales.column_name,
    x=missing_value_percentage_sales.percent_missing,
    hue=missing_value_percentage_sales.column_name,
    order=missing_value_percentage_sales.column_name
)

# Add a vertical line at x=50 (adjust as needed)
plt.axvline(x=75, color='red', linestyle='--', label='Threshold (75%)')

plt.title('Répartition des valeurs manquantes dans le dataset', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=9)
plt.ylabel('Features')
plt.legend()

plt.show()


# ### Elimination de colonnes (valeurs manquantes supérieures à 75 %)

# In[8]:


# Filtrer les colonnes avec un taux de valeurs manquantes inférieur ou égal à 75%
columns_to_keep = missing_data_percentage_sales[missing_data_percentage_sales <= 75].index

# Mettre à jour le DataFrame en gardant uniquement les colonnes sélectionnées
df_sales_short_1 = df_sales_clean[columns_to_keep]

print("Colonnes conservées :", list(columns_to_keep))
print("\nShape du Dataset après élimination des colonnes :", df_sales_short_1.shape)


# # DATAVIZ

# ### Modalités des variables ( moins de 10 modalités )

# In[9]:


# Combine object and numerical columns
columns_to_check = df_sales_short_1.select_dtypes(include=['object', 'int64', 'float64']).columns

columns_checked = []

# Iterate through each column and filter unique values with less than 10, excluding NaN
for col in columns_to_check:
    unique_values = df_sales_short_1[col].dropna().unique()  # Exclude NaN values
    if len(unique_values) < 10:
        print(f"Column: {col}")
        print(f"Unique Values: {unique_values}")
        print("-" * 50)
        columns_checked.append(col)

# Nous considérons à ce stade que les colonnes avec moins de 10 valeurs uniques sont des variables catégorielles


# In[10]:


for var_to_viz in columns_checked:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_sales_short_1, x=var_to_viz)
    plt.title(f'{var_to_viz}')
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


# ### Modalités des autres variables ( plus de 10 modalités )

# In[11]:


columns_investigated = columns_checked
df_sales_remaining = df_sales_short_1.drop(columns_investigated,axis=1)

columns_to_check = df_sales_remaining.select_dtypes(include=['object']).columns
columns_checked = []

# Iterate through each column and filter unique values with less than 10, excluding NaN
for col in columns_to_check:
    unique_values = df_sales_remaining[col].dropna().unique()  # Exclude NaN values
    if len(unique_values) > 10:
        print(f"Column: {col}")
        print(f"Unique Values: {unique_values}")
        print("-" * 50)
        columns_checked.append(col)

# Nous considérons à ce stade que les colonnes avec plus de 10 valeurs uniques sont 
# soit des variables numériques continues soit des variablles catégorielles à traiter car proposant trop de modes
# Nous sortons donc les valeurs uniques de ces colonnes pour les investiguer


# # PRÉSÉLECTION (VARIABLES EXPLICATIVES À ÉLIMINER)

# > La colonne 'idannonce' est un identifiant unique pour chaque annonce, elle n'est pas utile pour l'analyse
# 
# > La colonne 'annonce_exclusive' est une variable qui n'est pas utile pour l'analyse
# 
# > La colonne 'typedebien' et 'typedebien_lite' contiennent les mêmes informations; nous gardons la plus riche des deux : 'typedebien
# 
# > La colonne 'type_annonceur' offre une distribution de valeurs trop déséquilibrée
# 
# > La colonne 'duree_int' n'est pas interprétable (valeurs négatives, compréhension empirique)
# 
# > Les colonnes 'REG', 'DEP', 'IRIS', 'CODE_IRIS', 'TYP_IRIS_x', 'TYP_IRIS_y', 'GRD_QUART', 'UU2010' sont des colonnes contenant de l'information redondante, de plus nous créerons une nouvelle colonne pour le code postal, générée à partir des coordonnées géographiques
# 
# > nous gardons la colonne 'INSEE_COM' pour l'utiliser lors de la gestion des outliers

# In[12]:


df_sales_short_2 = df_sales_short_1.drop(columns=['idannonce', 'annonce_exclusive', 'typedebien_lite', 
                                                  'type_annonceur', 'categorie_annonceur',
                                                  'REG', 'DEP', 'IRIS', 'CODE_IRIS', 'TYP_IRIS_x', 'TYP_IRIS_y',
                                                  'nb_logements_copro',
                                                  'GRD_QUART', 'UU2010', 'duree_int'], axis=1)

df_sales_short_2.shape


# ### Liste colonnes restantes et vérification des NaNs

# In[13]:


missing_data_percentage_sales = df_sales_short_2.isna().sum()*100/len(df_sales_short_2)

missing_value_percentage_sales = pd.DataFrame({'column_name': df_sales_short_2.columns,
                                         'percent_missing': missing_data_percentage_sales,
                                         'dtypes':df_sales_short_2.dtypes}
                                         ).sort_values(by='percent_missing', ascending=False)

# Resetting the index to start from 1 for better readability
# and to match the original DataFrame's index
missing_value_percentage_sales.index = range(1, len(missing_value_percentage_sales) + 1)

display(missing_value_percentage_sales)


# # VARIABLES EXPLICATIVES À TRAITER

# In[14]:


# les variables porte_digicode, ascenseur et cave sont des variables binaires typées en 'object'
# nous les transofrmins en type boleeen
bool_columns = ['porte_digicode', 'cave', 'ascenseur']
for col in bool_columns:
    df_sales_short_2[col] = df_sales_short_2[col].astype(bool)

# vérification des types des colonnes converties
df_sales_short_2[bool_columns].dtypes


# ## Dicrétisations

# ### La variable "annee_construction" est transformée en variable catégorielle nominale

# In[15]:


# La variable "annee_construction" est transformée en variable catégorielle non ordinale :

# Définir les plages et les catégories (plages trouvées du le net comme étant celles correspondant à des ensembles cohérents)
bins = [float('-inf'), 1948, 1974, 1977, 1982, 1988, 2000, 2005, 2012, 2021, float('inf')]
labels = [
    "avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988",
    "1989-2000", "2001-2005", "2006-2012", "2013-2021", "après 2021"
]
df_sales_short_2['annee_construction'] = pd.cut(df_sales_short_2['annee_construction'], bins=bins, labels=labels, right=False)
# Vérification de la transformation
print(df_sales_short_2['annee_construction'].head())


# ### DPeL et ges_class

# In[16]:


# 1. Fonctions utilitaires

def clean_classe(val):
    """
    Nettoie et standardise les classes DPE/GES :
    - Vide, "Blank" ou "0" → np.nan
    - Conserve A→G, NS, VI
    - Sinon, extrait un code valide en début de chaîne via regex
    """
    # Cas manquant
    if pd.isna(val) or str(val).strip() in ["", "Blank", "0"]:
        return np.nan
    # Mise en majuscule et suppression des espaces
    s = str(val).strip().upper()
    # Acceptation stricte des codes connus
    if s in ["A","B","C","D","E","F","G","NS","VI"]:
        return s
    # Tentative de capture d'un code valide en début de chaîne
    m = re.match(r"^(NS|VI|[A-G])", s, re.IGNORECASE)
    return m.group(1).upper() if m else np.nan


def extract_principal(val):
    """
    Extrait la première source énergétique listée.
    Séparateurs gérés : ',', ';', '/', 'et'
    """
    # Cas manquant ou chaîne vide
    if pd.isna(val) or not str(val).strip():
        return np.nan
    # Split sur les séparateurs, on ne garde que la première partie
    parts = re.split(r"\s*(?:,|;|/|et)\s*", str(val).strip(), maxsplit=1)
    return parts[0] if parts else np.nan


# Configuration pour la normalisation de l'exposition
PATTERN_EXPO = r"(?i)\b(?:Nord(?:-Est|-Ouest)?|Sud(?:-Est|-Ouest)?|Est|Ouest|N|S|E|O)\b"
ORDRE_EXPO  = ["Nord","Est","Sud","Ouest"]
NORM_DIR    = {
    "N":"Nord","S":"Sud","E":"Est","O":"Ouest",
    "NORD":"Nord","SUD":"Sud","EST":"Est","OUEST":"Ouest",
    "NORD-EST":"Nord-Est","NORD-OUEST":"Nord-Ouest",
    "SUD-EST":"Sud-Est","SUD-OUEST":"Sud-Ouest"
}

def clean_exposition(val):
    """
    Nettoie et standardise la colonne exposition :
    - Détecte les mots-clés de multi-exposition → "Multi-exposition"
    - Extrait les points cardinaux via regex
    - Traduit et normalise selon NORM_DIR
    - Trie et déduplique selon ORDRE_EXPO
    """
    # Cas manquant ou valeur vide
    if pd.isna(val) or str(val).strip() in ["","0"]:
        return np.nan
    s   = str(val).strip()
    low = s.lower()
    # 1) Multi-exposition via mots-clés
    if any(kw in low for kw in ["traversant","multi","toutes","double expo","triple","360"]):
        return "Multi-exposition"
    # 2) Extraction des directions
    matches = re.findall(PATTERN_EXPO, s, flags=re.IGNORECASE)
    dirs = [
        NORM_DIR[m.upper().replace(" ","-")]
        for m in matches
        if m.upper().replace(" ","-") in NORM_DIR
    ]
    # 3) Tri et déduplication
    uniq = sorted(set(dirs), key=lambda d: ORDRE_EXPO.index(d.split("-")[0]))
    return "-".join(uniq) if uniq else np.nan


# 2. Application des fonctions sur df_sales_short_2

# 2.1 : Nettoyage des colonnes DPE et GES
for col in ("dpeL","ges_class"):
    if col in df_sales_short_2.columns:
        df_sales_short_2[col] = (
            df_sales_short_2[col]
            .apply(clean_classe)    # Applique clean_classe à chaque valeur
            .astype("object")       # Force le type chaîne pour les résultats
        )

# 2.2 : Extraction de l'énergie de chauffage principale
if "chauffage_energie" in df_sales_short_2.columns:
    df_sales_short_2["chauffage_energie_principal"] = (
        df_sales_short_2["chauffage_energie"]
            .apply(extract_principal)  # Garde la première source énergétique
            .astype("object")
    )
# Correction de l'encodage mal interprété (ex: Ã -> É)
df_sales_short_2["chauffage_energie_principal"] = (
    df_sales_short_2["chauffage_energie_principal"]
    .str.replace("Ã\x89", "É", regex=False)
)


# 2.3 : Nettoyage de la colonne exposition
if "exposition" in df_sales_short_2.columns:
    df_sales_short_2["exposition"] = (
        df_sales_short_2["exposition"]
            .apply(clean_exposition)   # Standardise les orientations
            .astype("object")
    )

# 3. Contrôles rapides pour valider le nettoyage

# Affiche un aperçu des premières lignes
display(df_sales_short_2[[
    "dpeL",
    "ges_class",
    "chauffage_energie_principal",
    "exposition"
]].head(10))

# Liste des valeurs uniques par colonne
print("Classes DPE :", df_sales_short_2["dpeL"].unique())
print("Classes GES :", df_sales_short_2["ges_class"].unique())
print("Énergies principales :", df_sales_short_2["chauffage_energie_principal"].unique())
print("Expositions :", df_sales_short_2["exposition"].unique())

# Supprimer la colonne temporaire APRÈS les contrôles
if "chauffage_energie_principal" in df_sales_short_2.columns:
    df_sales_short_2.drop(columns=["chauffage_energie_principal"], inplace=True)


# In[17]:


df_sales_short_2.isna().sum()
#df_sales_short_2.info()


# # VARIABLE CIBLE ET VARIABLES CORRÉLÉES À LA CIBLE

# ## Suppression des variables fortement corrélées au target (prix_m2_vente)

# La variable cible est "prix_m2_vente"

# > Certaines variables du csv de part leur nature pourraient être trop corrélées avec la variable cible  :
# > prix, mensualité, etc.

# In[18]:


# Calculer la matrice de corrélation
correlation_matrix = df_sales_short_2[['prix_bien', 'prix_m2_vente','mensualiteFinance']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de corrélation')
plt.show()


# > On va donc supprimer ces variables de la base de données :

# In[19]:


# On va donc supprimer ces variables de la base de données et renommer le DataFrame:
df_sales_short_3 = df_sales_short_2.drop(columns=['prix_bien', 'mensualiteFinance' ], axis=1)
df_sales_short_3.shape


# ## Visualisation de la distribution de la variable cible

# In[20]:


plt.figure(figsize=(8, 4))
sns.histplot(data=df_sales_short_3, x='prix_m2_vente', bins=30)
plt.title('Prix m2_vente Distribution')
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()


# > Il est problable qu'il y ait eu un problème de collectecte et la présence d'outliers, nous allons les traiter 

# # GESTION DES OUTLIERS ET DES VALEURS ABERRANTES

# ### Visualisation des distributions des variables numériques avec Boxplots 

# Avant traitements

# In[21]:


# Future colonne de regroupement pour les outliers
GROUP_COL    = 'INSEE_COM'  # colonne de regroupement

# Identification des colonnes numériques et exclusion des colonnes de coordonnées géographiques
def get_numeric_cols(data, group_col):
    """
    Retourne les colonnes numériques en excluant la future colonne de regroupement - qui est l'équivalent d'un code postal- pour les outliers à venir plus tard
    et les colonnes de coordonnées géographiques.
    """
    excluded_cols = ['mapCoordonneesLatitude', 'mapCoordonneesLongitude']
    return [
        col for col in data.select_dtypes(include='number').columns
    
        if col != group_col and col not in excluded_cols
    ]
numeric_cols = get_numeric_cols(df_sales_short_3, GROUP_COL)

# Visualisation des boxplots pour les colonnes restantes

# Nombre de colonnes par ligne
cols_per_row = 2

# Calcul du nombre de lignes nécessaires
num_cols = len(numeric_cols)
num_rows = math.ceil(num_cols / cols_per_row)

# Création des sous-graphiques
fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(12, 4 * num_rows))
axes = axes.flatten()  # Aplatir pour un accès plus simple

# Boucle pour tracer les boxplots
for i, col in enumerate(numeric_cols):
    df_sales_short_3.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f"Boxplot de la colonne '{col}'")

# Supprimer les axes inutilisés si le nombre de colonnes est impair
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Les boxplots montrent des distributions étonnantes.
# Par ailleur certaines variables semblent montrer des problèmes d'unités d'échelle
# Il s'agit des variables:    
# 'charges_copro'
# 'loyer_m2_median_n6'
# 'loyer_m2_median_n7'
# 'taux_rendement_n6' 
# 'taux_rendement_n7'
# 'nb_log_n6'
# 'nb_log_n7'
# 
# Nous allons essayer d'intépréter ces colonnes et voir si nous pouvons appliquer un traitement qui a du sens. Si ce n'est pas le cas, nous les éliminons ces colonnes. 
# 
# Dans l'ordre nous allons :
# - éliminer les valeurs aberrantes via la détection d'anomalies logiques
# - éliminer les valeurs aberrantes via les anomalies visuelles (suite aux boxplots)
# - traiter les valeurs extrêmes en créant des fonctions de détection des "outliers" et d'imputation par mediane de code INSEE.
# - traiter les problèmes d'unités d'échelle si encore apparents
# - éliminer les colonnes montrant encore des incohérences

# ## Anomalies

# ### Détection des anomalies logiques entre colonnes

# In[22]:


# Détection d'anomalies logiques dans les données suite aux boxplots 

# Création d'une colonne 'anomalie_logique' contenant True si une incohérence est détectée

# Copie pour éviter d'altérer la base brute directement
df_logic_check = df_sales_short_3.copy()

# Initialiser la colonne avec False
df_logic_check['anomalie_logique'] = False

# --- Règle 1 : nb_toilettes > nb_pieces (pas logique dans un logement classique)
mask_toilettes = df_logic_check['nb_toilettes'] > df_logic_check['nb_pieces']
df_logic_check.loc[mask_toilettes, 'anomalie_logique'] = True

# --- Règle 2 : surface trop petite (< 10 m²) ou démesurée (> 1000 m²)
mask_surface = (df_logic_check['surface'] < 10) | (df_logic_check['surface'] > 1000)
df_logic_check.loc[mask_surface, 'anomalie_logique'] = True

# --- Règle 3 : nb_etages = 0 alors que etage > 0 (impossible sans étage)
mask_etage = (df_logic_check['nb_etages'] == 0) & (df_logic_check['etage'] > 0)
df_logic_check.loc[mask_etage, 'anomalie_logique'] = True

# --- Règle 4 : logement neuf mais année de construction ancienne (avant 2000)
mask_neuf = (df_logic_check['logement_neuf'] == True) & (
    df_logic_check['annee_construction'].isin(["avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988", "1989-2000"])
)
df_logic_check.loc[mask_neuf, 'anomalie_logique'] = True

# --- Règle 5 : prix_m2_vente très bas ou nul (hors outlier déjà traité)
mask_prix = (df_logic_check['prix_m2_vente'] < 100)
df_logic_check.loc[mask_prix, 'anomalie_logique'] = True

# Résumé : Nombre total de lignes concernées
nb_anomalies = df_logic_check['anomalie_logique'].sum()
print(f"{nb_anomalies} lignes présentent au moins une anomalie logique.")

# Aperçu des premières anomalies détectées
display(df_logic_check[df_logic_check['anomalie_logique']].head(10))


# ### Détection des anomalies de saisie

# In[23]:


# L'idée ici est de borner les valeurs complètement aberrantes avant d'éliminer les valeurs extrêmes
# Colonnes à vérifier pour erreurs d’échelle
cols_suspectes = [
    'etage',
    'surface',
    'surface_terrain',
    'nb_pieces',
    'balcon',
    'bain',
    'dpeC',
    'nb_etages',
    'places_parking',
    'nb_toilettes',
    'charges_copro',
    'loyer_m2_median_n6',
    'nb_log_n6',
    'taux_rendement_n6',
    'loyer_m2_median_n7',
    'nb_log_n7',
    'taux_rendement_n7',
    'prix_m2_vente',
]

# Seuils définis de manière métier ou empirique
seuils_max = {
    'etage': 60,                         # étage > 60
    'surface': 2000,                    # surface > 2000 m²
    'surface_terrain': 500_000,          # terrain > 50 hectares
    'nb_pieces': 100,                    # plus de 100 pièces
    'balcon': 100,                         # plus de 100 balcons
    'bain': 20,                          # plus de 20 salles de bain
    'dpeC': 10000,                    # plus de 10 000 DPE C
    'nb_etages': 60,                    # plus de 60 étages
    'places_parking': 50,                # plus de 50 places de parking
    'nb_toilettes': 50,                 # plus de 50 toilettes
    'charges_copro': 10_000,             # charges mensuelles > 10k €
    'loyer_m2_median_n6': 500,               # loyer m2 > 500 €
    'nb_log_n6': 15000,                   # plus de 15000 logements
    'taux_rendement_n6': 1,             # taux de rendement > 100%
    'loyer_m2_median_n7': 500,               # loyer m2 > 500 €
    'nb_log_n7': 15000,                   # plus de 15000 logements
    'taux_rendement_n7': 1,             # taux de rendement > 100%
    'prix_m2_vente': 100_000,             # prix au m² > 100k €
}

seuils_min = {
    'etage': -3,                       # étage < -3
    'balcon': -1,                       # balcon < -1
    'dpeC': -1,                     # DPE C < -1
}

# Création d’un DataFrame résumé des cas problématiques
problemes = {}
mask_valeurs_improbables = pd.Series(False, index=df_sales_short_3.index)  # Initialiser le masque

for col in cols_suspectes:
    # Détection des valeurs au-dessus du seuil maximum
    if col in seuils_max:
        mask_above = df_sales_short_3[col] > seuils_max[col]
        mask_valeurs_improbables |= mask_above
        n_anormaux_max = mask_above.sum()
    else:
        n_anormaux_max = 0

    # Détection des valeurs en dessous du seuil minimum
    if col in seuils_min:
        mask_below = df_sales_short_3[col] < seuils_min[col]
        mask_valeurs_improbables |= mask_below
        n_anormaux_min = mask_below.sum()
    else:
        n_anormaux_min = 0

    # Ajouter au rapport si des valeurs aberrantes sont détectées
    if n_anormaux_max > 0 or n_anormaux_min > 0:
        problemes[col] = {
            'nb_anormaux_max': n_anormaux_max,
            'max_valeur': df_sales_short_3[col].max(),
            'nb_anormaux_min': n_anormaux_min,
            'min_valeur': df_sales_short_3[col].min()
        }

# Création d'un DataFrame pour le rapport
df_problemes = pd.DataFrame.from_dict(problemes, orient='index')

# Affichage du rapport
print("Rapport des valeurs improbables détectées :")
display(df_problemes)

# Nombre total de lignes identifiées comme improbables
nb_lignes_improbables = mask_valeurs_improbables.sum()
print(f"{nb_lignes_improbables} lignes contiennent des valeurs improbables.")


# ### Suppression des lignes concernées

# In[26]:


# --- Création d'un masque combiné pour les anomalies logiques et valeurs improbables ---

# Masque pour les anomalies logiques
mask_anomalies_logiques = df_logic_check['anomalie_logique']

# Combinaison des deux masques
mask_combined = mask_anomalies_logiques | mask_valeurs_improbables

# Nombre total de lignes identifiées comme problématiques
nb_lignes_problemes = mask_combined.sum()
print(f"{nb_lignes_problemes} lignes contiennent des anomalies logiques ou des valeurs improbables.")

# --- Suppression des lignes identifiées ---

# Suppression des lignes identifiées
df_sales_short_3 = df_sales_short_3[~mask_combined]

# Résumé : Nombre de lignes supprimées
print(f"{nb_lignes_problemes} lignes ont été supprimées en raison d'anomalies logiques ou de valeurs improbables.")

# Aperçu du DataFrame nettoyé
display(df_sales_short_3.head())


# ## Outliers

# ### Outliers Regression

# #### Imputation par mediane par code insee

# In[27]:


# SPLIT ET PARAMÈTRES

from sklearn.model_selection import train_test_split

#SPLIT 

train_data, test_data = train_test_split(df_sales_short_3, test_size=0.2, random_state=42)


#  Constantes et paramètres ─────────────
LOWER_PERC   = 0.001     # 1er dixième de percentile
UPPER_PERC   = 0.999     # dernier dixième de percentile
GROUP_COL    = 'INSEE_COM'  # colonne de regroupement
TARGET_COL   = 'prix_m2_vente'  # variable à prédire
OUTLIER_TAG  = -999      # valeur pour différencier les outliers


# In[28]:


# FONCTIONS ET CALCULS  ─────────────
# pas besoin de reinitialiser l'array numeric_cols déjà défini plus haut pour l'affichage des boxplots et exclutant les colonnes de coordonnées géographiques

def calculate_bounds(data, numeric_cols, lower_perc, upper_perc):
    """
    Calcule les bornes inférieures et supérieures pour chaque colonne numérique.
    """
    return {
        col: (
            data[col].quantile(lower_perc),
            data[col].quantile(upper_perc)
        )
        for col in numeric_cols
    }



def compute_medians(train_data, bounds, group_col):
    """
    Calcule les médianes par groupe (group_col) et globales
    UNIQUEMENT à partir du train).
    """
    group_meds = {
        col: train_data.groupby(group_col)[col].median()
        for col in bounds
    }
    global_meds = train_data[list(bounds)].median()
    return group_meds, global_meds




def mark_outliers(df, bounds, outlier_tag=OUTLIER_TAG):
    """
    Pour chaque colonne dans bounds :
      - crée col+'_outlier_flag' = 1 si en dehors des bornes
      - remplace la valeur outlier par outlier_tag
    """
    for col, (low, high) in bounds.items():
        mask = (df[col] < low) | (df[col] > high)
        df[f'{col}_outlier_flag'] = mask.astype(int)
        df.loc[mask, col] = outlier_tag
    return df



def clean_outliers(df, bounds, group_meds, global_meds, group_col, outlier_tag=OUTLIER_TAG):
    """
    Remplace dans df (train ou test) les tags outlier_tag par :
      - la médiane du groupe (via group_meds)
      - sinon la médiane globale (global_meds)
    """
    for col in bounds:
        mask = df[col] == outlier_tag
        # remplace uniquement les outliers taggés
        df.loc[mask, col] = (
            df.loc[mask, group_col]
                .map(group_meds[col])
                .fillna(global_meds[col])
                .astype(df[col].dtype) # Ajout d'un cast pour éviter les problèmes de type
        )
    return df

# Application des fonctions de nettoyage ----------------
## Bounds
bounds = calculate_bounds(train_data, numeric_cols, LOWER_PERC, UPPER_PERC)

## Médianes
group_medians, global_medians = compute_medians(train_data, bounds, GROUP_COL)

## Marquage des outliers
train_marked = mark_outliers(train_data, bounds)
test_marked  = mark_outliers(test_data,  bounds)

        ### masque de conservation 
mask_train_keep = train_marked[f'{TARGET_COL}_outlier_flag'] == 0
mask_test_keep  = test_marked[f'{TARGET_COL}_outlier_flag'] == 0

        ### application du filtre et suppression des outliers de la target
train_marked = train_marked[mask_train_keep]
test_marked  = test_marked[mask_test_keep]

# Calcul du nombre d'outliers identifiés par colonne avant leur remplacement
outlier_counts = {
    col: train_marked[f'{col}_outlier_flag'].sum()
    for col in bounds
}

## Nettoyage (remplacement des -999) avec les médianes du TRAIN
train_clean = clean_outliers(train_marked, bounds, group_medians, global_medians, GROUP_COL)
test_clean  = clean_outliers(test_marked, bounds, group_medians, global_medians, GROUP_COL)

# suppression des colonnes de marquage
train_clean.drop(columns=[f'{col}_outlier_flag' for col in bounds], inplace=True)
test_clean.drop(columns=[f'{col}_outlier_flag' for col in bounds], inplace=True)


# In[ ]:


# RECONSTITUTION DES JEUX X / y
#X_train = train_clean.drop(columns=[TARGET_COL])
#y_train = train_clean[TARGET_COL]
#X_test  = test_clean.drop(columns=[TARGET_COL])
#y_test  = test_clean[TARGET_COL]


# #### Après traitement outliers

# In[29]:


# Nombre de colonnes par ligne
cols_per_row = 2

# Calcul du nombre de lignes nécessaires
num_cols = len(bounds)
num_rows = math.ceil(num_cols / cols_per_row)

# Création des sous-graphiques
fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(12, 4 * num_rows))
axes = axes.flatten()  # Aplatir pour un accès plus simple

# Boucle pour tracer les boxplots
print("Visualisation des boxplots après traitement des outliers :")
for i, col in enumerate(bounds):
    train_clean.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f"Boxplot de la colonne '{col}' après traitement des outliers")

# Supprimer les axes inutilisés si le nombre de colonnes est impair
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# ### Outliers Serie temporelle

# #### Imputation par mediane par code insee

# In[ ]:


from sklearn.model_selection import train_test_split


# df_sales_short_3 = pd.concat([train_clean, test_clean], axis=0).reset_index(drop=True)

df_sales_short_3["date"] = pd.to_datetime(df_sales_short_3["date"], errors="coerce")
df_sales_short_3["Year"] = df_sales_short_3["date"].dt.year
df_sales_short_3["Month"] = df_sales_short_3["date"].dt.month

#SPLIT 

df_sales_short_3["split"] = df_sales_short_3["Year"].map(lambda x: "train_data_ST" if x < 2024 else "test_data_ST")
train_data_ST = df_sales_short_3[df_sales_short_3["split"] == "train_data_ST"]
test_data_ST  = df_sales_short_3[df_sales_short_3["split"] == "test_data_ST"]
df_sales_short_3 = df_sales_short_3.drop(columns='split')




#  Constantes et paramètres ─────────────
LOWER_PERC   = 0.001         # 1er percentile
UPPER_PERC   = 0.999         # 99e percentile
GROUP_COL    = 'INSEE_COM'  # colonne de regroupement
TARGET_COL   =  'prix_m2_vente'  # variable à prédire
OUTLIER_TAG  = -999         # valeur pour différencier les outliers


## Bounds


# FONCTIONS ET CALCULS  ─────────────
# pas besoin de reinitialiser l'array numeric_cols déjà défini plus haut pour l'affichage des boxplots et exclutant les colonnes de coordonnées géographiques

def calculate_bounds(data, numeric_cols, lower_perc, upper_perc):
    """
    Calcule les bornes inférieures et supérieures pour chaque colonne numérique.
    """
    return {
        col: (
            data[col].quantile(lower_perc),
            data[col].quantile(upper_perc)
        )
        for col in numeric_cols
    }



def compute_medians(train_data_ST, bounds_ST, group_col):
    """
    Calcule les médianes par groupe (group_col) et globales
    UNIQUEMENT à partir du train).
    """
    group_meds = {
        col: train_data_ST.groupby(group_col)[col].median()
        for col in bounds_ST
    }
    global_meds = train_data_ST[list(bounds_ST)].median()
    return group_meds, global_meds




def mark_outliers(df, bounds_ST, outlier_tag=OUTLIER_TAG):
    """
    Pour chaque colonne dans bounds_ST :
      - crée col+'_outlier_flag' = 1 si en dehors des bornes
      - remplace la valeur outlier par outlier_tag
    """
    for col, (low, high) in bounds_ST.items():
        mask = (df[col] < low) | (df[col] > high)
        df[f'{col}_outlier_flag'] = mask.astype(int)
        df.loc[mask, col] = outlier_tag
    return df



def clean_outliers(df, bounds_ST, group_meds, global_meds, group_col, outlier_tag=OUTLIER_TAG):
    """
    Remplace dans df (train ou test) les tags outlier_tag par :
      - la médiane du groupe (via group_meds)
      - sinon la médiane globale (global_meds)
    """
    for col in bounds_ST:
        mask = df[col] == outlier_tag
        # remplace uniquement les outliers taggés
        df.loc[mask, col] = (
            df.loc[mask, group_col]
                .map(group_meds[col])
                .fillna(global_meds[col])
                .astype(df[col].dtype) # Ajout d'un cast pour éviter les problèmes de type
        )
    return df

# Application des fonctions de nettoyage ----------------
## Bounds
bounds_ST = calculate_bounds(train_data_ST, numeric_cols, LOWER_PERC, UPPER_PERC)

## Médianes
group_medians_ST, global_medians_ST = compute_medians(train_data_ST, bounds_ST, GROUP_COL)

## Marquage des outliers
train_marked_ST = mark_outliers(train_data_ST, bounds_ST)
test_marked_ST  = mark_outliers(test_data_ST,  bounds_ST)

        ### masque de conservation 
mask_train_keep = train_marked_ST[f'{TARGET_COL}_outlier_flag'] == 0
mask_test_keep = test_marked_ST[f'{TARGET_COL}_outlier_flag'] == 0

        ### application du filtre et suppression des outliers de la target
train_marked_ST = train_marked_ST[mask_train_keep]
test_marked_ST  = test_marked_ST[mask_test_keep]

# Calcul du nombre d'outliers identifiés par colonne avant leur remplacement
outlier_counts_ST = {
    col: train_marked_ST[f'{col}_outlier_flag'].sum()
    for col in bounds_ST
}

## Nettoyage (remplacement des -999) avec les médianes du TRAIN
train_clean_ST = clean_outliers(train_marked_ST, bounds_ST, group_medians_ST, global_medians_ST, GROUP_COL)
test_clean_ST  = clean_outliers(test_marked_ST, bounds_ST, group_medians_ST, global_medians_ST, GROUP_COL)

# suppression des colonnes de marquage
train_clean_ST.drop(columns=[f'{col}_outlier_flag' for col in bounds_ST], inplace=True)
test_clean_ST.drop(columns=[f'{col}_outlier_flag' for col in bounds_ST], inplace=True)



# Vérification des outliers
print("Valeurs extrêmes détectées et remplacées :")
for col, count in outlier_counts_ST.items():
    print(f"Colonne '{col}: {count} outliers détectés et remplacés.")
# Vérification des valeurs extrêmes restantes
print("Valeurs extrêmes restantes :")
for col in bounds_ST:
    print(f"Colonne '{col}': {train_clean_ST[col].min()} à {train_clean_ST[col].max()}")



# #### Après traitement des outliers

# In[ ]:


# Nombre de colonnes par ligne
cols_per_row = 2

# Calcul du nombre de lignes nécessaires
num_cols = len(bounds_ST)
num_rows = math.ceil(num_cols / cols_per_row)

# Création des sous-graphiques
fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(12, 4 * num_rows))
axes = axes.flatten()  # Aplatir pour un accès plus simple

# Boucle pour tracer les boxplots
print("Visualisation des boxplots après traitement des outliers :")
for i, col in enumerate(bounds_ST):
    train_clean_ST.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f"Boxplot de la colonne '{col}' après traitement des outliers")

# Supprimer les axes inutilisés si le nombre de colonnes est impair
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# ## Visualisation de la distribution de la target

# In[ ]:


plt.figure(figsize=(8, 4))
sns.histplot(data=train_clean[prix_m2_vente], bins=150, kde=True)
plt.title('Distribution Prix m2_vente')
plt.xticks(rotation=45, fontsize=8)
plt.xlim(0, 20000)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()


# ## Visualisation de la distribution de la surface

# In[ ]:


plt.figure(figsize=(8, 4))
sns.histplot(data=train_clean['surface'], bins=150, kde=True)
plt.title('Distribution Surface')

# Ajuster les limites de l'axe X en fonction des valeurs minimales et maximales
plt.xlim(train_clean['surface'].min(), train_clean['surface'].max())

plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()


# # SAUVEGARDE DU DATASET

# ## Regression

# In[30]:


## Creation du Dataframe

# df_sales_short_3 = pd.concat([train_clean, test_clean], axis=0).reset_index(drop=True)

## paths
output_filepath = click.prompt('Enter the file path for the output preprocessed data', type=click.Path())



## stocker les datasets nettoyés dans le repertoire de travail et les nommer train_clean.csv et test_clean.csv

train_clean.to_csv(os.path.join(output_filepath, 'train_clean.csv'), sep=';', index=False)
test_clean.to_csv(os.path.join(output_filepath, 'test_clean.csv'), sep=';', index=False)






## Rappel des colonnes restantes
# print("Colonnes restantes dans le DataFrame :")
# print(df_sales_clean.columns)
# print(df_sales_clean.dtypes)
# print("\nShape du Dataset après élimination des colonnes :", df_sales_clean.shape)
print(train_clean.info())
print(test_clean.info())
display(df_sales_clean.head())
display(test_clean.head())



