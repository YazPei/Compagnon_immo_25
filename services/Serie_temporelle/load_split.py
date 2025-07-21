import os
import pandas as pd
import mlflow

def split_data(input_path, output_folder):
    """Charge le dataset global et le split en séries temporelles par cluster."""
    os.makedirs(output_folder, exist_ok=True)

    mlflow.set_experiment("ST-Split")
    with mlflow.start_run(run_name="split_clusters"):

        df = pd.read_csv(input_path, sep=';', parse_dates=['date'], index_col='date')
        df = df.dropna(subset=['cluster', 'prix_m2_vente'])
        df['cluster'] = df['cluster'].astype(int)

        cluster_list = sorted(df['cluster'].unique())
        mlflow.log_param("clusters", cluster_list)

        for cl in cluster_list:
            df_cl = df[df['cluster'] == cl].resample('M').prix_m2_vente.mean().dropna()
            out_path = os.path.join(output_folder, f"cluster_{cl}_series.csv")
            df_cl.to_csv(out_path, sep=';', index=True)
            mlflow.log_artifact(out_path)

        mlflow.log_metric("n_clusters", len(cluster_list))
        print(f"{len(cluster_list)} séries sauvegardées dans {output_folder}")

