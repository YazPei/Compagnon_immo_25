#!/usr/bin/env python
# coding: utf-8

# # Pipeline de Feature Selection et d'Analyse Avancée
# Ce notebook intègre la sélection de caractéristiques, l'optimisation d'hyperparamètres, l'entraînement final, la visualisation des résultats de CV et l'analyse des résidus.

# ## 1. Importation et Configuration

# In[ ]:


import re, os, pickle
import pandas as pd, numpy as np
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tabulate import tabulate

# Constantes
DATA_PATH = 'C:/Users/User/Downloads/drive-download-20250508T155351Z-1-001/'
CV_SETUP_FILE = 'cv_setup.pkl'
PARAM_FILE = '/mnt/data/param.py'
XGB_FINAL_FILE = '/mnt/data/XGBoost_final.py'
RESIDUS_FILE = '/mnt/data/Analyse_des_residus.py'
CV_RESULTS_FILE = '/mnt/data/cv_results.pkl'
N_SAMPLE = 150_000


# ## 2. Optimisation des hyperparamètres avec Optuna
# Chargement du code d'optimisation et exécution de l'étude Optuna pour XGBoost.

# In[ ]:


import os
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# 1) Chargement des données
DATA_PATH = 'C:/Users/User/Downloads/drive-download-20250508T155351Z-1-001/'
X = pd.read_csv(os.path.join(DATA_PATH, 'X_train_encoded.csv'), sep=';')
y = pd.read_csv(os.path.join(DATA_PATH, 'y_train.csv'), sep=';').values.ravel()

# 2) Sous-échantillon pour accélérer l’optimisation
X_sub, _, y_sub, _ = train_test_split(X, y, train_size=150_000, random_state=42)

# 3) Objectif Optuna
def objective(trial):
    params = {
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'random_state': 42,
        'verbosity': 0,
    }
    # split interne pour validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    y_pred = bst.predict(dval)
    mse = mean_squared_error(y_val, y_pred)          # on récupère le MSE
    rmse = np.sqrt(mse)                              # puis on en fait la racine
    return rmse

# 4) Lancement de l’étude
if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, timeout=60*30)  # 50 essais max ou 30 min

    print("→ Best RMSE:", study.best_value)
    print("→ Best params:", study.best_params)

```


# ## 3. Entraînement du modèle XGBoost final
# Utilisation des meilleurs hyperparamètres pour entraîner le modèle sur l'ensemble des données.

# In[ ]:


```python
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# 1) Chargement complet
DATA_PATH = 'C:/Users/User/Downloads/drive-download-20250508T155351Z-1-001/'
X_train = pd.read_csv(os.path.join(DATA_PATH, 'X_train_encoded.csv'), sep=';')
y_train = pd.read_csv(os.path.join(DATA_PATH, 'y_train.csv'), sep=';').values.ravel()
X_test  = pd.read_csv(os.path.join(DATA_PATH, 'X_test_encoded.csv'),  sep=';')
y_test  = pd.read_csv(os.path.join(DATA_PATH, 'y_test.csv'),  sep=';').values.ravel()

# 2) Préparation DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# 3) Paramètres optimaux trouvés
best_params = {
    'tree_method':      'gpu_hist',
    'device':           'cuda',
    'n_estimators':     1000,
    'max_depth':        12,
    'learning_rate':    0.024165459736487295,
    'subsample':        0.7827805784475923,
    'colsample_bytree': 0.9360622482907809,
    'gamma':            1.9279857202175867,
    'lambda':           5.317859770949592,
    'alpha':            0.0016284032243054373,
    'random_state':     42,
    'verbosity':        1,
}

# 4) Entraînement final
bst = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_params['n_estimators'],
    evals=[(dtrain, 'train')],
    verbose_eval=50
)

# 5) Évaluation test
y_pred = bst.predict(dtest)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"→ Test RMSE: {rmse:.3f}, R2: {r2:.3f}")

# 6) Importances
importances = bst.get_score(importance_type='gain')
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 importances (gain) :")
for feat, imp in sorted_imp:
    print(f"  {feat:30s} → {imp:.4f}")

# 7) Sauvegarde
bst.save_model("xgb_final.model")
print("\nModèle sauvegardé sous `xgb_final.model`")

```


# ## 4. Résultats de la validation croisée
# Chargement et affichage détaillé des performances de chaque pli.

# In[ ]:


import pickle
cv_results = pickle.load(open(CV_RESULTS_FILE, 'rb'))
df_cv = pd.DataFrame(cv_results)
display(df_cv)
plt.figure(figsize=(6,4))
df_cv['RMSE'].hist()
plt.title('Distribution de la RMSE par pli')
plt.xlabel('RMSE')
plt.ylabel('Fréquence')
plt.show()


# ## 5. Analyse avancée des résidus
# Histogramme, Q–Q plot et tests de normalité pour évaluer la distribution des résidus.

# In[ ]:


```python
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# --- 1) Définissez exactement le dossier où sont vos CSV ---
DATA_PATH = r"C:/Users/User/Downloads/drive-download-20250508T155351Z-1-001"

# --- 2) Chargez vos données d'entraînement et de test ---
X_train = pd.read_csv(os.path.join(DATA_PATH, 'X_train_encoded.csv'), sep=';')
y_train = pd.read_csv(os.path.join(DATA_PATH, 'y_train.csv'),         sep=';').values.ravel()
X_test  = pd.read_csv(os.path.join(DATA_PATH, 'X_test_encoded.csv'),  sep=';')
y_test  = pd.read_csv(os.path.join(DATA_PATH, 'y_test.csv'),          sep=';').values.ravel()

# --- 3) Rechargez le pipeline entraîné ---
with open('best_pipe.pkl', 'rb') as f:
    best_pipe = pickle.load(f)

# --- 4) Prédictions & résidus ---
y_pred_train = best_pipe.predict(X_train)
res_train    = y_train - y_pred_train

y_pred_test  = best_pipe.predict(X_test)
res_test     = y_test  - y_pred_test

# --- 5.1) Nuage de points résidus vs prédictions (train) ---
plt.figure(figsize=(6,4))
plt.scatter(y_pred_train, res_train, alpha=0.2, s=10)
plt.hlines(0, y_pred_train.min(), y_pred_train.max(), linestyles='--', linewidth=1)
plt.xlabel("Prédictions (train)")
plt.ylabel("Résidus (train)")
plt.title("Résidus vs Prédictions — entraînement")
plt.show()

# --- 5.2) Histogramme des résidus (test) ---
plt.figure(figsize=(6,4))
plt.hist(res_test, bins=50, density=True, alpha=0.7)
plt.xlabel("Résidu")
plt.ylabel("Densité")
plt.title("Distribution des résidus — test")
plt.show()

# --- 5.3) Q–Q plot des résidus (test) ---
plt.figure(figsize=(6,4))
st.probplot(res_test, dist="norm", plot=plt)
plt.title("Q–Q plot des résidus — test")
plt.show()

```

