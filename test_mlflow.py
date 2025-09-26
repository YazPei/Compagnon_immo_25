import os, mlflow
from dotenv import load_dotenv

load_dotenv()  # charge le fichier .env

# Vérification des variables d'environnement nécessaires
required_env_vars = ["DAGSHUB_USER", "DAGSHUB_TOKEN", "MLFLOW_TRACKING_URI"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Les variables d'environnement suivantes sont manquantes : {', '.join(missing_vars)}")

# Configuration des identifiants pour MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USER")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

# Configuration des URI pour MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))

print("✅ Connecté à MLflow avec l'URI :", mlflow.get_tracking_uri())

# Test de connexion et enregistrement d'un run
try:
    with mlflow.start_run(run_name="smoke-test"):
        mlflow.log_param("hello", "world")
        mlflow.log_metric("ping", 1.0)
    print("✅ Test MLflow réussi.")
except Exception as e:
    print("❌ Échec du test MLflow :", str(e))

