import mlflow
from mlflow_connect import configure_mlflow
def main():
   uri, user = configure_mlflow()
   print(f"test - Connecté à MLflow {uri} (user={user})")

with mlflow.start_run(run_name="docker-test"):
    mlflow.log_param("foo", "bar")
    mlflow.log_metric("acc", 0.98)
