"""
Service MLflow pour l'intégration avec l'API de prédiction.
Gère la configuration MLflow, le chargement des modèles depuis le registry,
et le logging des prédictions en temps réel.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class MLflowService:
    """Service pour intégrer MLflow dans l'API de prédiction."""

    def __init__(self):
        self.is_configured = False
        self.tracking_uri = None
        self.experiment_name = "api_predictions"
        self._configure_mlflow()

    def _configure_mlflow(self) -> bool:
        """Configurer MLflow avec les variables d'environnement."""
        try:
            # Charger les variables d'environnement
            load_dotenv()

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            username = os.getenv("MLFLOW_TRACKING_USERNAME")
            password = os.getenv("MLFLOW_TRACKING_PASSWORD")

            if not tracking_uri:
                logger.warning("MLFLOW_TRACKING_URI non défini, "
                              "MLflow non configuré")
                return False

            # Configurer MLflow
            mlflow.set_tracking_uri(tracking_uri)

            if username and password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = password

            # Tester la connexion
            try:
                mlflow.search_experiments()
                self.is_configured = True
                self.tracking_uri = tracking_uri
                logger.info("✅ MLflow configuré avec succès: "
                           f"{tracking_uri}")
                return True
            except Exception as e:
                logger.error(f"❌ Erreur de connexion MLflow: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Erreur lors de la configuration MLflow: {e}")
            return False

    def load_model_from_registry(self, model_name: str,
                                version: str = "latest") -> Optional[Any]:
        """Charger un modèle depuis le Model Registry MLflow."""
        if not self.is_configured:
            logger.warning("MLflow non configuré, "
                          "impossible de charger le modèle")
            return None

        try:
            if version == "latest":
                # Récupérer la dernière version de production
                client = mlflow.MlflowClient()
                versions = client.get_latest_versions(
                    model_name, stages=["Production"])
                if versions:
                    version = versions[0].version
                else:
                    # Si pas de version en production, prendre la dernière
                    model_versions = client.get_registered_model(
                        model_name).latest_versions
                    if model_versions:
                        version = model_versions[0].version
                    else:
                        logger.warning("Aucune version trouvée pour "
                                      f"{model_name}")
                        return None

            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("✅ Modèle {model_name} v{version} "
                       "chargé depuis MLflow")
            return model

        except Exception as e:
            logger.error("❌ Erreur chargement modèle {model_name} "
                        f"depuis MLflow: {e}")
            return None

    def log_prediction(self, model_name: str, model_version: str,
                      input_data: Dict[str, Any], prediction: Any,
                      prediction_time: float, success: bool = True) -> None:
        """Logger une prédiction dans MLflow."""
        if not self.is_configured:
            return

        try:
            with mlflow.start_run(
                experiment_id=self._get_or_create_experiment(),
                run_name=f"api_prediction_{model_name}"
            ) as run:

                # Log des paramètres
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_version", model_version)
                mlflow.log_param("input_features_count", len(input_data))

                # Log des métriques
                mlflow.log_metric("prediction_time_seconds", prediction_time)
                mlflow.log_metric("prediction_success",
                                 1.0 if success else 0.0)

                # Log de la prédiction (si numérique)
                if isinstance(prediction, (int, float)):
                    mlflow.log_metric("prediction_value", float(prediction))

                # Log des inputs importants (échantillonnage)
                for key, value in list(input_data.items())[:10]:
                    if isinstance(value, (int, float, str)):
                        mlflow.log_param(f"input_{key}",
                                        str(value)[:100])

                if not success:
                    mlflow.set_tag("prediction_status", "failed")
                else:
                    mlflow.set_tag("prediction_status", "success")

                logger.debug("✅ Prédiction loggée dans MLflow: "
                            f"{run.info.run_id}")

        except Exception as e:
            logger.error(f"❌ Erreur lors du logging MLflow: {e}")

    def _get_or_create_experiment(self) -> str:
        """Récupérer ou créer l'expérience MLflow pour les prédictions API."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id
            else:
                return mlflow.create_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"Erreur création expérience MLflow: {e}")
            # Retourner l'expérience par défaut
            return "0"

    def get_model_versions(self, model_name: str) -> List[str]:
        """Récupérer les versions disponibles d'un modèle."""
        if not self.is_configured:
            return []

        try:
            client = mlflow.MlflowClient()
            versions = client.get_registered_model(
                model_name).latest_versions
            return [v.version for v in versions] if versions else []
        except Exception as e:
            logger.error("Erreur récupération versions pour "
                        f"{model_name}: {e}")
            return []

    def get_health_status(self) -> Dict[str, Any]:
        """Status de santé du service MLflow."""
        return {
            "service": "mlflow_service",
            "configured": self.is_configured,
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
        }


# Instance globale du service MLflow
mlflow_service = MLflowService()
