import os
import click
import joblib
import mlflow
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

@click.command()
@click.option('--encoded-folder', prompt='Dossier des fichiers encodés', type=click.Path(exists=True))
@click.option('--use-gpu', is_flag=True, help="Activer le GPU si disponible")
def train_xgb_model(encoded_folder, use_gpu):
    mlflow.set_experiment("regression_pipeline")
    with mlflow.start_run(run_name="train_xgb"):
        X_train = pd.read_csv(os.path.join(encoded_folder, 'X_train.csv'), sep=';')
        y_train = pd.read_csv(os.path.join(encoded_folder, 'y_train.csv'), sep=';').values.ravel()
        X_test = pd.read_csv(os.path.join(encoded_folder, 'X_test.csv'), sep=';')
        y_test = pd.read_csv(os.path.join(encoded_folder, 'y_test.csv'), sep=';').values.ravel()

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'tree_method': 'gpu_hist' if use_gpu else 'auto',
                'random_state': 42
            }
            model = XGBRegressor(**params)
            cv = KFold(n_splits=3)
            score = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv).mean()
            return -score

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        mlflow.log_params(best_params)

        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)

        model_path = os.path.join(encoded_folder, 'xgb_model.joblib')
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print("Modèle XGBoost entraîné et sauvegardé.")

if __name__ == '__main__':
    train_xgb_model()

