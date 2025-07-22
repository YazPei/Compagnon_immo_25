import os
import argparse
import itertools
import pandas as pd
import numpy as np
import mlflow
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, normal_ad
from prophet import Prophet
import joblib
import glob

def filter_exog_by_corr(y, exog_df, threshold=0.05):
    corr_values = exog_df.corrwith(y).abs()
    return exog_df[corr_values[corr_values > threshold].index]

def apply_lag_if_correlated(df, target="prix_m2_vente", threshold=0.3):
    lagged_cols = []
    for col in df.columns:
        if col != target and df[col].isna().sum() == 0:
            corr = df[col].corr(df[target])
            if abs(corr) >= threshold:
                df[col] = df[col].shift(1)
                lagged_cols.append(col)
    df = df.dropna(subset=lagged_cols + [target])
    return df, lagged_cols


def adf_significant(series, cutoff=0.05):
    p_value = adfuller(series.dropna(), autolag='AIC')[1]
    return p_value <= cutoff

def get_diff_order(series, seasonal=False, s=12, max_d=2, cutoff=0.05):
    """
    Retourne l'ordre de différenciation nécessaire (0 à max_d) pour rendre la série stationnaire.
    """
    for d in range(0, max_d + 1):
        y_diff = series.copy()
        for _ in range(d):
            y_diff = y_diff.diff(s if seasonal else 1)
        if adf_significant(y_diff.dropna(), cutoff=cutoff):
            return d
    return max_d


def get_valid_seasonal_orders(D, s=12):
    return [(P, D, Q, s) for P in range(3) for Q in range(3)]


def get_exog_subsets(columns):
    return [list(combo) for i in range(1, len(columns) + 1)
            for combo in itertools.combinations(columns, i)]


def evaluate_model(model_result):
    ljung_pval = acorr_ljungbox(model_result.resid, lags=[12], return_df=True)['lb_pvalue'].iloc[0]
    normal_pval = normal_ad(model_result.resid)[1]
    return {
        'aic': model_result.aic,
        'bic': model_result.bic,
        'llf': model_result.llf,
        'ljung_pvalue': ljung_pval,
        'normal_pvalue': normal_pval
    }


def all_exog_significant(model_result, exog_vars):
    return all(model_result.pvalues.get(var, 1) <= 0.05 for var in exog_vars)


def try_sarimax_grid(y, exog_df, d, D, s):
    best_model = None
    best_metrics = None
    best_params = None

    exog_subsets = get_exog_subsets(exog_df.columns)  # ✅ ici !

    for p, q in itertools.product(range(3), repeat=2):
        for P, D_seas, Q, s_val in get_valid_seasonal_orders(D, s):
            for exog_subset in exog_subsets:  # ✅ et ici on les utilise bien
                try:
                    exog = exog_df[exog_subset] if exog_subset else None
                    model = sm.tsa.SARIMAX(
                        endog=y,
                        exog=exog,
                        order=(p, d, q),
                        seasonal_order=(P, D_seas, Q, s_val),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    result = model.fit(disp=False)

                    metrics = evaluate_model(result)
                    all_significant = all_exog_significant(result, exog_subset)

                    with mlflow.start_run(nested=True):
                        mlflow.log_param("order", (p, d, q))
                        mlflow.log_param("seasonal_order", (P, D_seas, Q, s_val))
                        mlflow.log_param("exog_vars", exog_subset)
                        mlflow.log_param("all_exog_significant", all_significant)
                        for key, val in metrics.items():
                            mlflow.log_metric(key, val)
                        mlflow.set_tag("valid_model", all_significant)
                        mlflow.set_tag("tested_model", True)

                    if not all_significant:
                        continue

                    if (metrics['ljung_pvalue'] > 0.05 and
                        metrics['normal_pvalue'] > 0.05 and
                        (best_model is None or metrics['aic'] < best_metrics['aic'])):
                        best_model, best_metrics, best_params = result, metrics, {
                            "order": (p, d, q),
                            "seasonal_order": (P, D_seas, Q, s_val),
                            "exog_vars": exog_subset
                        }

                except Exception as e:
                    mlflow.set_tag("error", str(e))
                    continue

    return best_model, best_metrics, best_params



def fallback_prophet(df, cluster_id, output_folder, suffix=""):
    print(f"[Cluster {cluster_id}] ❌ Aucun SARIMAX valide, fallback vers Prophet.")
    prophet_df = df.reset_index()[["date", "prix_m2_vente"]].rename(columns={"date": "ds", "prix_m2_vente": "y"})
    model = Prophet()
    model.fit(prophet_df)
    model_path = os.path.join(output_folder, f"best/cluster_{cluster_id}_prophet{suffix}.pkl")
    joblib.dump(model, model_path)
    mlflow.set_tag("fallback", "prophet")


def train_all_clusters(input_folder, output_folder, suffix="", save_split=False):
    os.makedirs(os.path.join(output_folder, "best"), exist_ok=True)
    mlflow.set_experiment("ST-SARIMAX-AutoSearch")

    for file in os.listdir(input_folder):
        if not (file.endswith(".csv") and "cluster_" in file):
            continue

        cluster_id = file.split("_")[1]
        df = pd.read_csv(os.path.join(input_folder, file), sep=";", parse_dates=["date"]).sort_values("date")
        df = df.set_index("date")
        df, lagged_cols = apply_lag_if_correlated(df)
        y = df["prix_m2_vente"]
        
        if save_split:
            split_idx = int(len(df) * 0.8)
            df_train = df.iloc[:split_idx]
            df_test = df.iloc[split_idx:]

            train_path = os.path.join(output_folder, f"train_cluster_{cluster_id}.csv")
            test_path  = os.path.join(output_folder, f"test_cluster_{cluster_id}.csv")

            df_train.to_csv(train_path, sep=";")
            df_test.to_csv(test_path, sep=";")
        
        print(f"\n🌀 [Cluster {cluster_id}] Analyse débutée")

        # Calcul des ordres
        d = get_diff_order(y, seasonal=False)
        D = get_diff_order(y, seasonal=True, s=12)
        s = 12

        # Cas 1 : ADF KO → Prophet
        if d > 2 or D > 2:
            with mlflow.start_run(run_name=f"cluster_{cluster_id}"):
                mlflow.log_param("ADF_d", d)
                mlflow.log_param("ADF_D", D)
                mlflow.log_param("ADF_fallback", True)
                mlflow.log_param("lagged_exog", lagged_cols)
                mlflow.set_tag("reason", "adf_failed")
                fallback_prophet(df, cluster_id, output_folder, suffix=suffix)
            continue

        with mlflow.start_run(run_name=f"cluster_{cluster_id}"):
            mlflow.log_param("ADF_d", d)
            mlflow.log_param("ADF_D", D)
            mlflow.log_param("ADF_s", s)
            mlflow.log_param("lagged_exog", lagged_cols)
            mlflow.log_param("ADF_fallback", False)

            # Sélection des exogènes
            exog_vars = [col for col in df.columns if col not in ["prix_m2_vente", "cluster"]]
            exog_df = df[exog_vars]
            mlflow.log_param("exog_final", exog_df.columns.tolist())

        #  Cas 2 : exogènes vides après lag
            if exog_df.empty:
                print(f"[Cluster {cluster_id}] ! Aucune variable exogène après filtrage !")
                mlflow.set_tag("no_exog_after_lag", True)
                mlflow.set_tag("reason", "no_exog_remaining")
                fallback_prophet(df, cluster_id, output_folder, suffix=suffix)
                continue

            # ✅ Entraînement SARIMAX
            best_model, best_metrics, best_params = try_sarimax_grid(y, exog_df, d, D, s)

            if best_model:
                model_path = os.path.join(output_folder, f"best/cluster_{cluster_id}_sarimax{suffix}.pkl")
                res_path = os.path.join(output_folder, f"best/forecast_cluster_{cluster_id}.csv")
                joblib.dump(best_model, model_path)
                mlflow.log_artifact(model_path)

                for key, val in best_metrics.items():
                    mlflow.log_metric(key, val)
                for key, val in best_params.items():
                    mlflow.log_param(key, val)

                print(f"✅ [Cluster {cluster_id}] SARIMAX validé et sauvegardé.")
            else:
        # Cas 3 : tous les modèles échouent → Prophet
                mlflow.set_tag("reason", "no_valid_sarimax")
                fallback_prophet(df, cluster_id, output_folder, suffix=suffix)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="", help="Suffixe à ajouter au nom du modèle")
    parser.add_argument("--save-split", action="store_true", help="Sauvegarder les fichiers train/test")
    args = parser.parse_args()
    train_all_clusters(args.input_folder, args.output_folder, suffix=args.suffix, save_split=args.save_split)




