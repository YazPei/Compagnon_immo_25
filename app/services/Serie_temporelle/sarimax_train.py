import os

import mlflow
import pandas as pd
import statsmodels.api as sm
from utils import adf_test, save_model


def train_sarimax_models(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    mlflow.set_experiment("ST-SARIMAX")

    for file in os.listdir(input_folder):
        if file.endswith(".csv") and "cluster_" in file:
            cluster_id = file.split("_")[1]
            path = os.path.join(input_folder, file)
            df = pd.read_csv(path, sep=";", parse_dates=["date"], index_col="date")
            df = df.asfreq("M")
            series = df["prix_m2_vente"].dropna()

            with mlflow.start_run(run_name=f"sarimax_cluster_{cluster_id}"):
                # Stationnarité test
                adf_result = adf_test(series)
                mlflow.log_metric("ADF_stat", adf_result["statistic"])
                mlflow.log_metric("ADF_pvalue", adf_result["pvalue"])

                # Modèle SARIMAX simple (à adapter manuellement au besoin)
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
                    results = model.fit(disp=False)

                    y_pred = results.fittedvalues
                    df["prediction"] = y_pred

                    rmse = ((series - y_pred) ** 2).mean() ** 0.5
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("aic", results.aic)
                    mlflow.log_param("order", order)
                    mlflow.log_param("seasonal_order", seasonal_order)

                    # Sauvegarde
                    save_path = os.path.join(
                        output_folder, f"sarimax_model_cluster_{cluster_id}.pkl"
                    )
                    save_model(results, save_path)
                    mlflow.log_artifact(save_path)

                    # Prévision future
                    forecast = results.get_forecast(steps=6)
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

                except Exception as e:
                    print(f"[Cluster {cluster_id}] ERREUR d'entraînement : {e}")
