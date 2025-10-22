"""
Module d'entraînement unifié pour les modèles immobiliers.
Consolide les fonctionnalités d'entraînement LGBM et SARIMAX.
"""

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import mlflow
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
import statsmodels.api as sm

from ..utils import adf_test, save_model
from ..evaluation.metrics import calculate_regression_metrics, calculate_time_series_metrics


class ImmoTrainer:
    """Classe principale pour l'entraînement des modèles immobiliers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train_lgbm_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                        X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[LGBMRegressor, Dict[str, float]]:
        """Entraîne un modèle LGBM avec optimisation Optuna."""

        mlflow.set_experiment("regression_pipeline")

        with mlflow.start_run(run_name="train_lgbm"):
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "random_state": 42,
                }
                model = LGBMRegressor(**params)
                cv = KFold(n_splits=3)
                score = cross_val_score(
                    model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=cv
                ).mean()
                return -score

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_params
            mlflow.log_params(best_params)

            model = LGBMRegressor(**best_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = calculate_regression_metrics(y_test, y_pred)
            mlflow.log_metrics(metrics)

            return model, metrics

    def train_sarimax_models(self, input_folder: str, output_folder: str) -> Dict[str, Any]:
        """Entraîne des modèles SARIMAX pour chaque cluster."""

        os.makedirs(output_folder, exist_ok=True)
        mlflow.set_experiment("ST-SARIMAX")
        results = {}

        for file in os.listdir(input_folder):
            if file.endswith(".csv") and "cluster_" in file:
                cluster_id = file.split("_")[1]
                path = os.path.join(input_folder, file)
                df = pd.read_csv(path, sep=";", parse_dates=["date"], index_col="date")
                df = df.asfreq("M")
                series = df["prix_m2_vente"].dropna()

                with mlflow.start_run(run_name=f"sarimax_cluster_{cluster_id}"):
                    # Test de stationnarité
                    adf_result = adf_test(series)
                    mlflow.log_metric("ADF_stat", adf_result["statistic"])
                    mlflow.log_metric("ADF_pvalue", adf_result["pvalue"])

                    # Modèle SARIMAX simple
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, 12)

                    try:
                        model = sm.tsa.SARIMAX(
                            series,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        fitted_model = model.fit(disp=False)

                        y_pred = fitted_model.fittedvalues
                        df["prediction"] = y_pred

                        metrics = calculate_time_series_metrics(series, y_pred)
                        mlflow.log_metrics(metrics)
                        mlflow.log_metric("aic", fitted_model.aic)
                        mlflow.log_param("order", order)
                        mlflow.log_param("seasonal_order", seasonal_order)

                        # Sauvegarde
                        save_path = os.path.join(
                            output_folder, f"sarimax_model_cluster_{cluster_id}.pkl"
                        )
                        save_model(fitted_model, save_path)
                        mlflow.log_artifact(save_path)

                        # Prévision future
                        forecast = fitted_model.get_forecast(steps=6)
                        forecast_df = forecast.summary_frame()
                        forecast_df.to_csv(
                            os.path.join(
                                output_folder, f"forecast_cluster_{cluster_id}.csv"
                            ),
                            sep=";",
                        )
                        mlflow.log_artifact(
                            os.path.join(
                                output_folder, f"forecast_cluster_{cluster_id}.csv"
                            )
                        )

                        results[cluster_id] = {
                            "model": fitted_model,
                            "metrics": metrics,
                            "forecast": forecast_df
                        }

                    except Exception as e:
                        self.logger.error(f"[Cluster {cluster_id}] Erreur d'entraînement : {e}")
                        results[cluster_id] = {"error": str(e)}

        return results

    def run_training(self, data_path: str) -> Dict[str, Any]:
        """Pipeline d'entraînement principal."""

        self.logger.info("Début de l'entraînement...")

        # Chargement des données
        # (Implémentation selon les besoins spécifiques)

        # Entraînement selon le type de modèle
        model_type = self.config.get("model_type", "lgbm")

        if model_type == "lgbm":
            # Logique d'entraînement LGBM
            pass
        elif model_type == "sarimax":
            # Logique d'entraînement SARIMAX
            pass

        return {"status": "completed"}
