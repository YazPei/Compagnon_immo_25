import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

            # Sauvegarde du CSV
            csv_path = os.path.join(model_folder, "global_sarimax_metrics.csv")
            df_metrics.to_csv(csv_path, sep=';', index=False)
            mlflow.log_artifact(csv_path)

            # Affichage console
            print(df_metrics)

            # Barplot RMSE par cluster
            plt.figure(figsize=(8, 5))
            sns.barplot(data=df_metrics, x='cluster', y='rmse', palette='crest')
            plt.title("RMSE par cluster (SARIMAX)")
            plt.xlabel("Cluster")
            plt.ylabel("RMSE")
            plt.tight_layout()

            plot_path = os.path.join(model_folder, "rmse_by_cluster.png")
            plt.savefig(plot_path)
            plt.close()

            # Log dans MLflow
            mlflow.log_artifact(plot_path)
        else:
            print("Aucun fichier de prévision trouvé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer les performances SARIMAX par cluster")
    parser.add_argument("--model-folder", type=str, required=True, help="Dossier contenant les fichiers forecast")
    args = parser.parse_args()
    
    evaluate_models(args.model_folder)

