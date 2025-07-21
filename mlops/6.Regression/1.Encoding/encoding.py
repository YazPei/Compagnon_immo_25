import os
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow

@click.command()
@click.option('--data-path', prompt='Chemin vers df_cluster.csv', help='Fichier CSV complet avec colonne split')
@click.option('--output', prompt='Dossier de sortie', help='Où sauvegarder les fichiers encodés')
def encode_data(data_path, output):
    mlflow.set_experiment("regression_pipeline")
    with mlflow.start_run(run_name="encoding"):
        print(f"Chargement depuis {data_path}")
        df = pd.read_csv(data_path, sep=';', parse_dates=['date'])

        df = df.dropna(subset=['prix_m2_vente'])
        df["date"] = pd.to_datetime(df["date"])

        train = df[df['split'] == 'train']
        test  = df[df['split'] == 'test']
        X_train = train.drop(columns=['prix_m2_vente'])
        y_train = train[['prix_m2_vente']]
        X_test  = test.drop(columns=['prix_m2_vente'])
        y_test  = test[['prix_m2_vente']]

        os.makedirs(output, exist_ok=True)
        X_train.to_csv(os.path.join(output, 'X_train.csv'), sep=';', index=False)
        y_train.to_csv(os.path.join(output, 'y_train.csv'), sep=';', index=False)
        X_test.to_csv(os.path.join(output, 'X_test.csv'), sep=';', index=False)
        y_test.to_csv(os.path.join(output, 'y_test.csv'), sep=';', index=False)

        mlflow.log_artifact(os.path.join(output, 'X_train.csv'))
        mlflow.log_artifact(os.path.join(output, 'X_test.csv'))

        print("Encodage terminé et sauvegardé.")

if __name__ == '__main__':
    encode_data()

