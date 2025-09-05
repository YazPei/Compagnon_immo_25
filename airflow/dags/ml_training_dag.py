from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Ajout du chemin pour les scripts
sys.path.append('/opt/airflow/scripts')

default_args = {
    'owner': 'compagnon-immo',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_dag',
    default_args=default_args,
    description='Pipeline d\'entraînement ML pour estimation immobilière',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'training', 'estimation'],
)

def train_model(**context):
    """Entraîne le modèle d'estimation."""
    print("🤖 Début de l'entraînement du modèle...")
    
    import mlflow
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    import joblib
    import os
    
    # Configuration MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    mlflow.set_experiment("compagnon-immo-training")
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Génération de données d'exemple (à remplacer par vraies données)
        np.random.seed(42)
        n_samples = 5000
        
        data = pd.DataFrame({
            'surface': np.random.normal(70, 25, n_samples),
            'nb_pieces': np.random.choice([1, 2, 3, 4, 5, 6], n_samples),
            'nb_chambres': np.random.choice([0, 1, 2, 3, 4], n_samples),
            'etage': np.random.choice(range(0, 11), n_samples),
            'annee_construction': np.random.choice(range(1950, 2024), n_samples),
            'code_postal_num': np.random.choice(range(75001, 75021), n_samples)
        })
        
        # Prix simulé avec un peu de logique
        data['prix'] = (
            data['surface'] * np.random.normal(8000, 1000, n_samples) +
            data['nb_pieces'] * np.random.normal(15000, 5000, n_samples) +
            (2024 - data['annee_construction']) * np.random.normal(-500, 200, n_samples) +
            np.random.normal(0, 50000, n_samples)
        )
        
        # Séparation features/target
        features = ['surface', 'nb_pieces', 'nb_chambres', 'etage', 'annee_construction', 'code_postal_num']
        X = data[features]
        y = data['prix']
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraînement
        params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Logging MLflow
        mlflow.log_params(params)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # Sauvegarde du modèle
        model_path = "/opt/airflow/models/model_latest.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        print(f"✅ Modèle entraîné - R²: {r2:.3f}, MAE: {mae:.0f}")
        
        return {"r2_score": r2, "mae": mae, "model_path": model_path}

def validate_model(**context):
    """Valide le modèle entraîné."""
    print("🔍 Validation du modèle...")
    
    # Récupération des résultats de la tâche précédente
    metrics = context['task_instance'].xcom_pull(task_ids='train_model')
    
    r2_threshold = 0.7
    mae_threshold = 100000
    
    if metrics['r2_score'] >= r2_threshold and metrics['mae'] <= mae_threshold:
        print(f"✅ Modèle validé - R²: {metrics['r2_score']:.3f}")
        return True
    else:
        print(f"❌ Modèle rejeté - R²: {metrics['r2_score']:.3f}")
        raise ValueError("Modèle ne respecte pas les critères de qualité")

def deploy_model(**context):
    """Déploie le modèle validé."""
    print("🚀 Déploiement du modèle...")
    
    # Ici vous pouvez ajouter la logique de déploiement
    # Par exemple, copier le modèle vers l'API
    import shutil
    
    source = "/opt/airflow/models/model_latest.joblib"
    destination = "/opt/airflow/models/model_production.joblib"
    
    shutil.copy2(source, destination)
    print("✅ Modèle déployé en production")

# Définition des tâches
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

notify_task = BashOperator(
    task_id='notify_completion',
    bash_command='echo "🎉 Pipeline ML terminé avec succès!"',
    dag=dag,
)

# Définition des dépendances
train_task >> validate_task >> deploy_task >> notify_task