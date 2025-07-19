#2
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import mlflow


def run_decomposition(input_folder, output_folder, suffix=""):
    os.makedirs(output_folder, exist_ok=True)
    mlflow.set_experiment("ST-Decomposition")

    for file in os.listdir(input_folder):
        if file.endswith(f"{suffix}.csv") and "cluster_" in file:
            cluster_id = file.split("_")[1]
            series_path = os.path.join(input_folder, file)
            df = pd.read_csv(series_path, sep=';', parse_dates=['date'], index_col='date')
            
            with mlflow.start_run(run_name=f"decomp_cluster_{cluster_id}{suffix}"):
                for model in ['additive', 'multiplicative']:
                    try:
                        # Décomposition + visu
                        decomp = seasonal_decompose(df['prix_m2_vente'], model=model, period=12)
                        fig = decomp.plot()
                        fig.suptitle(f"Décomposition {model} - Cluster {cluster_id}")
                        save_path = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}{suffix}.png")
                        plt.tight_layout()
                        plt.savefig(save_path)
                        plt.close()
                        mlflow.log_artifact(save_path)

                        # Composantes
                        trend = decomp.trend.dropna()
                        seasonal = decomp.seasonal.dropna()
                        resid = decomp.resid.dropna()

                        # CSV
                        trend_path = os.path.join(output_folder, f"trend_{model}_cluster_{cluster_id}{suffix}.csv")
                        seasonal_path = os.path.join(output_folder, f"seasonal_{model}_cluster_{cluster_id}{suffix}.csv")
                        resid_path = os.path.join(output_folder, f"resid_{model}_cluster_{cluster_id}{suffix}.csv")

                        trend.to_csv(trend_path, sep=';')
                        seasonal.to_csv(seasonal_path, sep=';')
                        resid.to_csv(resid_path, sep=';')

                        mlflow.log_artifact(trend_path)
                        mlflow.log_artifact(seasonal_path)
                        mlflow.log_artifact(resid_path)

                        mlflow.log_metric(f"{model}_trend_mean", trend.mean())
                        mlflow.log_metric(f"{model}_resid_std", resid.std())
                        mlflow.log_metric(f"{model}_resid_skew", resid.skew())

                    except Exception as e:
                        print(f"[Cluster {cluster_id}] Erreur {model} : {e}")
                        continue

# Bloc d'exécution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Décomposition saisonnière des séries par cluster")
    parser.add_argument("--input-folder", type=str, required=True, help="Dossier des séries temporelles")
    parser.add_argument("--output-folder", type=str, required=True, help="Dossier de sortie des résultats")
    parser.add_argument("--suffix", type=str, default="", help="Suffixe à utiliser pour filtrer les fichiers")

    args = parser.parse_args()
    run_decomposition(args.input_folder, args.output_folder, args.suffix)

