#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import warnings
import click
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor
import joblib
import mlflow
import json

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import psutil
    N_PHYS = psutil.cpu_count(logical=False) or os.cpu_count() or 1
except Exception:
    N_PHYS = os.cpu_count() or 1
    
TARGET = "prix_m2_vente"

# ---------- Sanitizer des noms de colonnes (LightGBM-safe) ----------
def sanitize_feature_names(cols):
    """
    Conserve uniquement [A-Za-z0-9_], remplace le reste par '_'.
    Évite les collisions en suffixant __2, __3, ...
    Force un préfixe 'f_' si le nom commence par un chiffre.
    """
    safe = []
    used = set()
    mapping = {}
    for c in cols:
        orig = str(c)
        new = re.sub(r'[^A-Za-z0-9_]', '_', orig)          # remplace caractères spéciaux
        new = re.sub(r'_+', '_', new).strip('_')            # compresse / trim underscores
        if new == "" or new[0].isdigit():
            new = f"f_{new}" if new != "" else "f"
        base = new
        k = 1
        while new in used:
            k += 1
            new = f"{base}__{k}"
        used.add(new)
        mapping[orig] = new
        safe.append(new)
    return safe, mapping

def apply_mapping(df, mapping):
    return df.rename(columns=mapping)

# --------------------------------------------------------------------

@click.command()
@click.option("--encoded-folder", default="data/encoded", show_default=True,
              help="Dossier contenant X_train.csv, y_train.csv, X_test.csv, y_test.csv.")
@click.option("--experiment", default="regression_pipeline", show_default=True,
              help="Nom d'expérience MLflow.")
@click.option("--tuner", type=click.Choice(["none", "random"], case_sensitive=False),
              default="none", show_default=True,
              help="Stratégie d'entraînement: 'none' (baseline) ou 'random' (RandomizedSearchCV léger).")
@click.option("--n-iter", default=30, show_default=True,
              help="Nombre d'itérations si --tuner=random.")
@click.option("--cv", default=5, show_default=True,
              help="Nombre de folds CV pour la recherche aléatoire.")
@click.option("--random-state", default=42, show_default=True, help="Seed.")
@click.option("--mem-mode", type=click.Choice(["auto","row","col"], case_sensitive=False),
              default="auto", show_default=True,
              help="Heuristique LightGBM (auto) ou forcer row/col-wise.")
def main(encoded_folder, experiment, tuner, n_iter, cv, random_state, mem_mode):
    X_train_path = os.path.join(encoded_folder, "X_train.csv")
    y_train_path = os.path.join(encoded_folder, "y_train.csv")
    X_test_path  = os.path.join(encoded_folder, "X_test.csv")
    y_test_path  = os.path.join(encoded_folder, "y_test.csv")

    # Chargement
    X_train = pd.read_csv(X_train_path, sep=';', low_memory=False)
    y_train = pd.read_csv(y_train_path, sep=';', low_memory=False)[TARGET].astype(float)
    X_test  = pd.read_csv(X_test_path,  sep=';', low_memory=False)
    y_test  = pd.read_csv(y_test_path,  sep=';', low_memory=False)[TARGET].astype(float)

    # Alignement structurel (avant sanitization)
    assert set(X_train.columns) == set(X_test.columns), "Colonnes train/test encodées différentes."
    X_train = X_train[X_test.columns]  # même ordre

    # ---- Sanitize des noms de colonnes (évite l'erreur LightGBM) ----
    safe_cols, mapping = sanitize_feature_names(X_train.columns.tolist())
    X_train.columns = safe_cols
    X_test = X_test.rename(columns=mapping)[safe_cols]

    # Log du mapping pour audit
    mapping_path = os.path.join(encoded_folder, "feature_name_map.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"train_lgbm_{tuner}"):
        mlflow.log_artifact(mapping_path)

        # Modèle de base (robuste)
        base_params = dict(
            random_state=random_state,
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=N_PHYS, 
            
        )
        if mem_mode.lower() == "row":
            base_params["force_row_wise"] = True
        elif mem_mode.lower() == "col":
            base_params["force_col_wise"] = True
        model = LGBMRegressor(**base_params)
        best_params = base_params.copy()

        if tuner.lower() == "random":
            param_distributions = {
                "num_leaves":        [20, 24, 31, 40, 48, 60],
                "learning_rate":     [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3],
                "n_estimators":      [200, 400, 600, 800, 1000, 1200, 1500],
                "max_depth":         [-1, 4, 6, 8, 10, 12],
                "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha":         [0.0, 0.01, 0.1, 0.5, 1.0],
                "reg_lambda":        [0.0, 0.01, 0.1, 0.5, 1.0],
                "min_child_samples": [5, 10, 20, 30, 50],
            }

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=int(n_iter),
                scoring="r2",
                cv=int(cv),
                n_jobs=1,
                verbose=1,
                random_state=random_state,
                refit=True
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model.fit(X_train, y_train)

        # Hold-out
        y_pred = model.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        print(f"R²  : {r2:.4f}")
        print(f"RMSE: {rmse:,.2f}")
        mlflow.log_metric("r2_test", r2)
        mlflow.log_metric("rmse_test", rmse)

        # CV stratifiée (sur quantiles de y) — optionnelle
        try:
            y_strat = pd.qcut(y_train, q=5, duplicates='drop', labels=False)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=skf.split(X_train, y_strat),
                                        scoring='r2', n_jobs=1)
            print(f"Scores R² par fold : {cv_scores}")
            print(f"Score R² moyen (CV) : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            mlflow.log_metric("cv_r2_mean", float(np.mean(cv_scores)))
            mlflow.log_metric("cv_r2_std",  float(np.std(cv_scores)))
        except Exception as e:
            print(f"[WARN] CV stratifiée non calculée : {e}")

        # Sauvegardes
        best_model_path  = os.path.join(encoded_folder, "best_lgbm_model.pkl")
        best_params_path = os.path.join(encoded_folder, "best_lgbm_params.pkl")
        joblib.dump(model, best_model_path)
        joblib.dump(best_params, best_params_path)
        # Compat avec la stage 'analyse'
        joblib.dump(model, os.path.join(encoded_folder, "lgbm_model.joblib"))

        mlflow.log_artifact(best_model_path)
        mlflow.log_artifact(best_params_path)
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        print("✅ Entraînement terminé. Modèle & params sauvegardés.")

if __name__ == "__main__":
    main()