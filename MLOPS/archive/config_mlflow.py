import mlflow

def setup_mlflow(experiment_name: str = "compagnon_immo") -> None:
    """Configure MLflow tracking URI et définit l'expérience"""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

