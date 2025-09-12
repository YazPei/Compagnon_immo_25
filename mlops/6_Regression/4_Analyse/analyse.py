import os
import click
import joblib
import pandas as pd
import mlflow
from utils import compute_metrics, print_metrics, plot_residuals, shap_summary_plot

@click.command()
@click.option('--encoded-folder', prompt='Dossier des fichiers encodés', type=click.Path(exists=True))
@click.option('--model', type=click.Choice(['lightgbm', 'xgboost']), prompt='Modèle à analyser')
def analyse_model(encoded_folder, model):
    mlflow.set_experiment("regression_pipeline")
    with mlflow.start_run(run_name=f"analyse_{model}"):
        model_path = os.path.join(encoded_folder, f'{model}_model.joblib')
        X_test_path = os.path.join(encoded_folder, 'X_test.csv')
        y_test_path = os.path.join(encoded_folder, 'y_test.csv')

        model_obj = joblib.load(model_path)
        X_test = pd.read_csv(X_test_path, sep=';')
        y_test = pd.read_csv(y_test_path, sep=';').values.ravel()

        y_pred = model_obj.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        print_metrics(metrics)
        plot_residuals(y_test, y_pred)
        shap_summary_plot(model_obj, X_test)
        # Nouveau : sauvegarde le SHAP summary dans un PNG
        shap_summary_plot(model_obj, X_test, out_path="exports/reg/shap_summary.png")
        # Log le fichier image dans MLflow
        mlflow.log_artifact("exports/reg/shap_summary.png")

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(X_test_path)

