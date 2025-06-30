def replace_minus1(X):
    import numpy as np
    return np.where(X == -1, 0, X)

def plus1(X):
    return X + 1

def invert11(X):
    return 1 - X

def cyclical_encode(df):
    import numpy as np
    import pandas as pd
    # Si df est une DataFrame à une seule colonne (ex: ["date"]), on la gère
    if isinstance(df, pd.DataFrame):
        if "date" in df.columns:
            date = pd.to_datetime(df["date"])
        else:
            raise ValueError("La colonne 'date' est attendue pour l'encodage cyclique.")
    else:
        # Si c'est une Series
        date = pd.to_datetime(df)
    # Création des features cycliques
    res = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * date.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * date.dt.month / 12),
        "dow_sin":   np.sin(2 * np.pi * date.dt.weekday / 7),
        "dow_cos":   np.cos(2 * np.pi * date.dt.weekday / 7),
    })
    return res 