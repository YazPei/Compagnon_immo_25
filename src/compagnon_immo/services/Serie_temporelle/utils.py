import pickle

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series):
    """Effectue le test d'ADF sur une série temporelle."""
    result = adfuller(series.dropna())
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "n_obs": result[3],
        "crit_values": result[4],
    }


def save_model(model, path):
    """Sauvegarde un modèle statsmodels avec pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
