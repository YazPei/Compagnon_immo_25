import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

def evaluate_models(model_folder):
    mlflow.set_experiment("ST-Evaluation")
    metrics = []

    with mlflow.start_run(run_name="evaluation_sarimax"):

        for file in os.listdir(model_folder):
            if file.startswith("forecast_cluster_") and file.endswith(".csv"):
                cluster_id = file.split("_")[2].split(".")[0]
                df = pd.read_csv(os.path.join(model_folder, file), sep=';')

                if 'mean' in df.columns:
                    pred = df['mean']
                    if 'mean_se' in df.columns:
                        se = df['mean_se']
                        rmse = (se ** 2).mean() ** 0.5
                    else:
                        rmse = pred.std()

                    metrics.append({
                        'cluster': cluster_id,
                        'rmse': rmse,
                        'forecast_mean': pred.mean()
                    })

        if metrics:
            df_metrics = pd.DataFrame(metrics)
            df_metrics.to_csv(os.path.join(model_folder, "global_sarimax_metrics.csv"), sep=';', index=False)
            mlflow.log_artifact(os.path.join(model_folder, "global_sarimax_metrics.csv"))
            print(df_metrics)
        else:
            print("Aucun fichier de prévision trouvé.")

