# mlflow_connect.py
import os
from pathlib import Path

def load_env():
    # facultatif: charger .env si présent
    try:
        from dotenv import load_dotenv
        env_path = Path(".") / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        pass

def configure_mlflow():
    load_env()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    user = os.getenv("MLFLOW_TRACKING_USERNAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")  # DagsHub token

    missing = [k for k,v in {
        "MLFLOW_TRACKING_URI": uri,
        "MLFLOW_TRACKING_USERNAME": user,
        "MLFLOW_TRACKING_PASSWORD": token,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Variables manquantes: {', '.join(missing)}. "
            "Créez un .env local ou exportez-les dans l'environnement."
        )

    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    return uri, user

if __name__ == "__main__":
    uri, who = configure_mlflow()
    print(f"MLflow prêt pour {who} → {uri}")
