# ---------------------------------------
# Imports
# ---------------------------------------
import os
import time
from numba import prange, njit
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px

from category_encoders import TargetEncoder

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans

from lightgbm import LGBMRegressor

import optuna
from optuna.integration import OptunaSearchCV
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

import shap
import joblib

# ---------------------------------------
# Chargement et nettoyage initial\ n# ---------------------------------------
csv_path = '/Users/loick.d/Documents/Datascientest/Github immo/MAR25_BDS_Compagnon_Immo/df_sales_clean.csv'
chunks = pd.read_csv(csv_path, sep=';', chunksize=100_000, low_memory=False)
df = pd.concat(chunks, ignore_index=True)

# Spatial join pour récupérer le vrai code postal
shp_path = '/Users/loick.d/Documents/Datascientest/Github immo/MAR25_BDS_Compagnon_Immo/contours-codes-postaux.geojson'
postal_shapes = gpd.read_file(shp_path).to_crs(epsg=4326)
points = gpd.GeoDataFrame(
    df,
    geometry=[Point(lon, lat) for lon, lat in zip(
        df.mapCoordonneesLongitude,
        df.mapCoordonneesLatitude
    )],
    crs='EPSG:4326'
)
joined = gpd.sjoin(points,
                   postal_shapes[['codePostal','geometry']],
                   how='left', predicate='within')
postal_series = joined['codePostal'].loc[~joined.index.duplicated(keep='first')]
df['codePostal'] = (
    postal_series
    .reindex(df.index)
    .fillna('inconnu')
    .astype(str)
)

# Regroupement hybride postal/départemental
counts = df['codePostal'].str.zfill(5).value_counts()
freqs = set(counts[counts>=10].index)
def regroup_hybride(z):
    if z in freqs:
        return z
    if z.isdigit() and len(z)==5:
        return z[:2]
    return 'inconnu'
df['zone_mixte'] = df['codePostal'].str.zfill(5).apply(regroup_hybride).astype('category')

# KMeans clustering sur agrégats
agg = df.groupby('zone_mixte')['prix_m2_vente'].agg(
    prix_m2_mean='mean', prix_m2_std='std',
    prix_m2_max='max', prix_m2_min='min'
)
agg['prix_m2_cv'] = agg['prix_m2_std'] / agg['prix_m2_mean']
agg = agg.replace([np.inf,-np.inf], np.nan).dropna()
X_scaled = StandardScaler().fit_transform(agg)
kmeans = KMeans(n_clusters=4, random_state=0)
agg['cluster'] = kmeans.fit_predict(X_scaled)
df = df.merge(agg[['cluster']], left_on='zone_mixte', right_index=True, how='left')
df['cluster'] = df['cluster'].fillna(-1).astype(int).astype('category')

# Visualisations (facultatives)
sns.pairplot(agg.reset_index(), vars=['prix_m2_mean','prix_m2_std','prix_m2_max','prix_m2_min','prix_m2_cv'], hue='cluster', diag_kind='kde', plot_kws={'alpha':0.6,'s':40})
plt.show()

# ---------------------------------------
# 0️⃣ Génération des features géographiques
# ---------------------------------------
lat = np.radians(df['mapCoordonneesLatitude'])
lon = np.radians(df['mapCoordonneesLongitude'])
df['x_geo'] = np.cos(lat) * np.cos(lon)
df['y_geo'] = np.cos(lat) * np.sin(lon)
df['z_geo'] = np.sin(lat)

# ---------------------------------------
# 1️⃣ Déclarations des colonnes pour le pipeline
# ---------------------------------------
target_variable = 'prix_m2_vente'
ordinal_cols   = ['ges_class','dpeL','logement_neuf','nb_pieces','bain','eau','nb_toilettes','balcon']
onehot_cols    = ['typedebien','typedetransaction','chauffage_mode','chauffage_energie_principal','cluster']
target_cols    = ['etage','nb_etages','exposition','chauffage_energie','chauffage_systeme','date']
numeric_cols   = ['surface','surface_terrain','dpeC','places_parking','charges_copro','loyer_m2_median_n6','nb_log_n6','taux_rendement_n6','loyer_m2_median_n7','nb_log_n7','taux_rendement_n7']
geo_cols       = ['x_geo','y_geo','z_geo']
year_col       = ['annee_construction']
year_order = ['après 2021','2013-2021','2006-2012','2001-2005','1989-2000','1983-1988','1978-1982','1975-1977','1948-1974','avant 1948']

# ---------------------------------------
# 2️⃣ Pipelines individuels
# ---------------------------------------
ordinal_pipeline = SkPipeline([('impute',SimpleImputer(strategy='most_frequent')), ('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)), ('scale',StandardScaler())])
onehot_pipeline  = SkPipeline([('impute',SimpleImputer(strategy='most_frequent')), ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
target_pipeline  = SkPipeline([('impute',SimpleImputer(strategy='most_frequent')), ('target',TargetEncoder()), ('scale',StandardScaler())])
numeric_pipeline = SkPipeline([('impute',SimpleImputer(strategy='median')), ('scale',StandardScaler())])
year_rank_pipeline = SkPipeline([
    ('impute',SimpleImputer(strategy='most_frequent')), 
    ('ord',OrdinalEncoder(categories=[year_order],dtype=int,handle_unknown='use_encoded_value',unknown_value=-1)), 
    ('replace',FunctionTransformer(lambda x: np.where(x==-1,5,x))),
    ('plus1',FunctionTransformer(lambda x: x+1)),
    ('invert',FunctionTransformer(lambda x: 11-x)),
    ('scale',StandardScaler())
])
geo_pipeline     = SkPipeline([('scale',StandardScaler())])

# ---------------------------------------
# 3️⃣ Assembleur final
# ---------------------------------------
preprocessor = ColumnTransformer([
    ('ord',      ordinal_pipeline,   ordinal_cols),
    ('ohe',      onehot_pipeline,    onehot_cols),
    ('year',     year_rank_pipeline, year_col),
    ('tar',      target_pipeline,    target_cols),
    ('num',      numeric_pipeline,   numeric_cols),
    ('geo',      geo_pipeline,       geo_cols),
])

pipeline = SkPipeline([
    ('preproc', preprocessor),
    ('model',   LGBMRegressor(boosting_type='gbdt', n_jobs=1, verbose=-1, random_state=42))
])

# ---------------------------------------
# 4️⃣ Split initial train/test
# ---------------------------------------
X = df.drop(columns=[target_variable])
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------
# 5️⃣ Sous-échantillon stratifié (20%)
# ---------------------------------------
y_train_bins = pd.qcut(y_train, q=10, duplicates='drop')
X_train_small, _, y_train_small, _ = train_test_split(
    X_train, y_train, train_size=0.2, stratify=y_train_bins, random_state=42
)

# ---------------------------------------
# 6️⃣ Optimisation Optuna
# ---------------------------------------
param_distributions = {
    'model__num_leaves':       optuna.distributions.IntDistribution(20,60),
    'model__learning_rate':    optuna.distributions.FloatDistribution(0.01,0.3,log=True),
    'model__n_estimators':     optuna.distributions.IntDistribution(100,1500),
    'model__max_depth':        optuna.distributions.IntDistribution(3,12),
    'model__subsample':        optuna.distributions.FloatDistribution(0.6,1.0),
    'model__colsample_bytree': optuna.distributions.FloatDistribution(0.6,1.0),
    'model__reg_alpha':        optuna.distributions.FloatDistribution(0.0,1.0),
    'model__reg_lambda':       optuna.distributions.FloatDistribution(0.0,1.0),
}
optuna_search = OptunaSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    cv=5,
    n_trials=20,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
optuna_search.fit(X_train_small, y_train_small)

# ---------------------------------------
# 7️⃣ Évaluation & détection de data leakage
# ---------------------------------------
# Hold-out
pipeline_fit = optuna_search.best_estimator_
pipeline_fit.fit(X_train, y_train)
r2_holdout = pipeline_fit.score(X_test, y_test)
print(f"R² hold-out test : {r2_holdout:.4f}")

# TimeSeriesSplit
X_np, y_np = X.to_numpy(), y.to_numpy()
tscv = TimeSeriesSplit(n_splits=5)
scores_ts = cross_val_score(pipeline_fit, X_np, y_np, cv=tscv, scoring='r2')
print(f"R² TimeSeriesSplit CV : {scores_ts.mean():.4f} ± {scores_ts.std():.4f}")

# KFold & StratifiedKFold sur small train
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(pipeline_fit, X_train_small.to_numpy(), y_train_small.to_numpy(), cv=kf, scoring='r2')
print(f"R² KFold CV (small train) : {scores_kf.mean():.4f} ± {scores_kf.std():.4f}")

y_small_bins = pd.qcut(y_train_small, q=5, duplicates='drop', labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(pipeline_fit, X_train_small.to_numpy(), y_train_small.to_numpy(), cv=skf.split(X_train_small, y_small_bins), scoring='r2')
print(f"R² StratifiedKFold CV (small train) : {scores_skf.mean():.4f} ± {scores_skf.std():.4f}")
