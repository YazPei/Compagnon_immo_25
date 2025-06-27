# Generated from notebook: Part-5.ipynb
import mlflow
mlflow.set_experiment('companion_immo')


# ---- CODE CELL ----
import os
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib

with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# !pip install pandas
# !pip install matplotlib

with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
import tensorflow as tf
print(tf.__version__)

with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
## Paths
# folder_path_M = '/Users/maximehenon/Documents/GitHub/MAR25_BDS_Compagnon_Immo/'
# folder_path_Y = 'C:/Users/charl/OneDrive/Documents/Yasmine/DATASCIENTEST/FEV25-BDS-COMPAGNON'
folder_path_C = '../data/processed/Sales'
#folder_path_L= '/Users/loick.d/Documents/Datascientest/Github immo/MAR25_BDS_Compagnon_Immo/'

# Reload datas
# X_test_encoded = pd.read_csv(os.path.join(folder_path_M, 'X_test_encoded.csv'), sep=';')
# X_train_encoded = pd.read_csv(os.path.join(folder_path_M, 'X_train_encoded.csv'), sep=';')
# # y_train_clean = pd.read_csv(os.path.join(folder_path_M, 'y_train_clean.csv'), sep=';')
# y_train = pd.read_csv(os.path.join(folder_path_M, 'y_train.csv'), sep=';')

# X_test_encoded = pd.read_csv(os.path.join(folder_path_Y, 'X_test_encoded.csv'), sep=';')
# X_train_encoded = pd.read_csv(os.path.join(folder_path_Y, 'X_train_encoded.csv'), sep=';')
# # y_train_clean = pd.read_csv(os.path.join(folder_path_Y, 'y_train_clean.csv'), sep=';')
# y_train = pd.read_csv(os.path.join(folder_path_Y, 'y_train.csv'), sep=';')

X_test_encoded = pd.read_csv(os.path.join(folder_path_C, 'X_test_encoded.csv'), sep=';')
X_train_encoded = pd.read_csv(os.path.join(folder_path_C, 'X_train_encoded.csv'), sep=';')
# y_train_clean = pd.read_csv(os.path.join(folder_path_C, 'y_train_clean.csv'), sep=';')
y_train = pd.read_csv(os.path.join(folder_path_C, 'y_train.csv'), sep=';')

# X_test_encoded = pd.read_csv(os.path.join(folder_path_L, 'X_test_encoded.csv'), sep=';')
# X_train_encoded = pd.read_csv(os.path.join(folder_path_L, 'X_train_encoded.csv'), sep=';')
# # y_train_clean = pd.read_csv(os.path.join(folder_path_L, 'y_train_clean.csv'), sep=';')
# y_train = pd.read_csv(os.path.join(folder_path_L, 'y_train.csv'), sep=';')

total_rows = 4147030  # le nombre réel de lignes
sample_size = int(0.1 * total_rows)

X_test_encoded = pd.read_csv(os.path.join(folder_path_C, 'X_test_encoded.csv'), sep=';',nrows=sample_size)
X_train_encoded = pd.read_csv(os.path.join(folder_path_C, 'X_train_encoded.csv'), sep=';',nrows=sample_size)
y_test = pd.read_csv(os.path.join(folder_path_C, 'y_test.csv'), sep=';',nrows=sample_size)
y_train = pd.read_csv(os.path.join(folder_path_C, 'y_train.csv'), sep=';',nrows=sample_size)


with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# liste des features par RFE
X_train_encoded = X_train_encoded[['surface', 'surface_terrain', 'dpeC', 'places_parking', 'charges_copro', 'duree_int', 'loyer_m2_median_n6', 'nb_log_n6', 'taux_rendement_n6', 'loyer_m2_median_n7', 'nb_log_n7', 'taux_rendement_n7', 'bain_outlier_flag', 'duree_int_outlier_flag', 'eau_outlier_flag', 'etage_outlier_flag', 'loyer_m2_median_n6_outlier_flag', 'loyer_m2_median_n7_outlier_flag', 'nb_log_n7_outlier_flag', 'nb_pieces_outlier_flag', 'nb_toilettes_outlier_flag', 'places_parking_outlier_flag', 'surface_outlier_flag', 'surface_terrain_outlier_flag', 'taux_rendement_n6_outlier_flag', 'taux_rendement_n7_outlier_flag', 'ges_class', 'dpeL', 'logement_neuf', 'nb_pieces', 'bain', 'eau', 'nb_toilettes', 'exposition', 'chauffage_energie', 'chauffage_systeme', 'date', 'nb_etages', 'typedebien_a', 'typedebien_an', 'typedebien_m', 'typedebien_mn', 'typedetransaction_vp', 'cave', 'annee_construction_1948-1974', 'annee_construction_2001-2005', 'annee_construction_2006-2012', 'annee_construction_2013-2021', 'annee_construction_MISSING', 'annee_construction_après 2021', 'annee_construction_avant 1948', 'porte_digicode', 'ascenseur', 'chauffage_mode_Individuel', 'chauffage_mode_Individuel, Central', 'chauffage_mode_MISSING', 'x_geo', 'y_geo', 'z_geo']]
X_test_encoded = X_test_encoded[['surface', 'surface_terrain', 'dpeC', 'places_parking', 'charges_copro', 'duree_int', 'loyer_m2_median_n6', 'nb_log_n6', 'taux_rendement_n6', 'loyer_m2_median_n7', 'nb_log_n7', 'taux_rendement_n7', 'bain_outlier_flag', 'duree_int_outlier_flag', 'eau_outlier_flag', 'etage_outlier_flag', 'loyer_m2_median_n6_outlier_flag', 'loyer_m2_median_n7_outlier_flag', 'nb_log_n7_outlier_flag', 'nb_pieces_outlier_flag', 'nb_toilettes_outlier_flag', 'places_parking_outlier_flag', 'surface_outlier_flag', 'surface_terrain_outlier_flag', 'taux_rendement_n6_outlier_flag', 'taux_rendement_n7_outlier_flag', 'ges_class', 'dpeL', 'logement_neuf', 'nb_pieces', 'bain', 'eau', 'nb_toilettes', 'exposition', 'chauffage_energie', 'chauffage_systeme', 'date', 'nb_etages', 'typedebien_a', 'typedebien_an', 'typedebien_m', 'typedebien_mn', 'typedetransaction_vp', 'cave', 'annee_construction_1948-1974', 'annee_construction_2001-2005', 'annee_construction_2006-2012', 'annee_construction_2013-2021', 'annee_construction_MISSING', 'annee_construction_après 2021', 'annee_construction_avant 1948', 'porte_digicode', 'ascenseur', 'chauffage_mode_Individuel', 'chauffage_mode_Individuel, Central', 'chauffage_mode_MISSING', 'x_geo', 'y_geo', 'z_geo']]


with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
# print(type(X_train_encoded))  # Should be (num_samples, 60)
# print(type(X_test_encoded))  # Should be (num_samples, 60)
X_train_encoded['porte_digicode'] = X_train_encoded['porte_digicode'].astype(int)
X_train_encoded['ascenseur'] = X_train_encoded['ascenseur'].astype(int)
X_train_encoded['cave'] = X_train_encoded['cave'].astype(int)
print(X_train_encoded.dtypes)

with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorflow as tf



# Construction du modèle
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(59,)))
model.add(Dense(64, activation='relu'))                     
model.add(Dense(1))                                         

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')



# Entraînement du modèle
model.fit(X_train_encoded, y_train, epochs=10, batch_size=10, validation_split=0.2)

loss = model.evaluate(X_test_encoded, y_test)
print(f'Loss sur les données de test: {loss}')

# Prédiction
predictions = model.predict(X_test_encoded)
print(predictions)

with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
residus = predictions[0] - y_test['prix_m2_vente']
residus = pd.DataFrame(residus)
# display(residus.describe())
residus.shape

with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)


# ---- CODE CELL ----
plt.plot((y_test['prix_m2_vente'].min(), y_test['prix_m2_vente'].max()), (0, 0), lw=3, color='red');
plt.plot(predictions, y_test['prix_m2_vente'], 'o', color='Green', markersize=1)
plt.scatter(y_test['prix_m2_vente'], residus['prix_m2_vente'], color='black', s=1)
plt.xlabel('Prédictions du modèle')
plt.xlim(0,y_train['prix_m2_vente'].max()) #  y_train.max())


with mlflow.start_run(run_name='Part-5'):
    # Log your metrics or parameters
    mlflow.log_metric('metric_name', value)

