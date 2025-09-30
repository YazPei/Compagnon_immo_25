import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def run_decomposition(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    mlflow.set_experiment("ST-Decomposition")

    for file in os.listdir(input_folder):
        if file.endswith(".csv") and "cluster_" in file:
            cluster_id = file.split("_")[1]
            series_path = os.path.join(input_folder, file)
            df = pd.read_csv(
                series_path, sep=";", parse_dates=["date"], index_col="date"
            )

            with mlflow.start_run(run_name=f"decomp_cluster_{cluster_id}"):
                for model in ["additive", "multiplicative"]:
                    try:
                        decomp = seasonal_decompose(
                            df["prix_m2_vente"], model=model, period=12
                        )
                        fig = decomp.plot()
                        fig.suptitle(f"Décomposition {model} - Cluster {cluster_id}")
                        save_path = os.path.join(
                            output_folder,
                            f"decomposition_{model}_cluster_{cluster_id}.png",
                        )
                        plt.tight_layout()
                        plt.savefig(save_path)
                        plt.close()
                        mlflow.log_artifact(save_path)
                    except Exception as e:
                        print(f"[Cluster {cluster_id}] Erreur {model} : {e}")
                        continue


s
