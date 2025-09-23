import os, mlflow
from dotenv import load_dotenv

load_dotenv()  # charge le fichier .env

os.environ["COMPAGNON_SECRET_YAZ"] = os.getenv("COMPAGNON_SECRET_YAZ")
os.environ["COMPAGNON_SECRET_KETSIA"] = os.getenv("COMPAGNON_SECRET_KETSIA")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USER")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))

print("Connected to:", mlflow.get_tracking_uri())

with mlflow.start_run(run_name="smoke-test"):
    mlflow.log_param("hello", "world")
    mlflow.log_metric("ping", 1.0)

