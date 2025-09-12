"""
Module pour vérifier la santé des services et dépendances.
"""

import asyncio
import time
import logging
from typing import Dict, Any
import httpx
import redis
from sqlalchemy import text

from app.api.config.settings import settings
from app.api.db.database import engine
from app.api.services.ml_service import ml_service

logger = logging.getLogger(__name__)


class HealthChecker:
    """Vérificateur de santé des services."""

    def __init__(self):
        self.start_time = time.time()

    async def check_database(self) -> Dict[str, Any]:
        """Vérifie la santé de la base de données."""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            return {
                "status": "healthy",
                "message": "Base de données accessible",
                "response_time_ms": 0  # TODO: mesurer le temps réel
            }
        except Exception as e:
            logger.error(f"Erreur base de données : {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Erreur base de données : {str(e)}",
                "response_time_ms": 0
            }

    async def check_mlflow(self) -> Dict[str, Any]:
        """Vérifie la connexion à MLflow."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health")

                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "message": "MLflow accessible",
                        "response_time_ms": 0
                    }
                else:
                    return {
                        "status": "degraded",
                        "message": f"MLflow status: {response.status_code}",
                        "response_time_ms": 0
                    }
        except Exception as e:
            logger.error(f"Erreur MLflow : {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"MLflow inaccessible : {str(e)}",
                "response_time_ms": 0
            }

    async def check_redis(self) -> Dict[str, Any]:
        """Vérifie la connexion à Redis."""
        try:
            r = redis.from_url(settings.REDIS_URL, decode_responses=True)
            r.ping()

            return {
                "status": "healthy",
                "message": "Redis accessible",
                "response_time_ms": 0
            }
        except Exception as e:
            logger.error(f"Erreur Redis : {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Redis inaccessible : {str(e)}",
                "response_time_ms": 0
            }

    async def check_models(self) -> Dict[str, Any]:
        """Vérifie l'état des modèles ML."""
        try:
            models_status = await ml_service.get_models_status()

            if models_status.get("models_loaded", 0) > 0:
                return {
                    "status": "healthy",
                    "message": f"{models_status['models_loaded']} modèles chargés",
                    "details": models_status
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Aucun modèle chargé",
                    "details": models_status
                }
        except Exception as e:
            logger.error(f"Erreur modèles : {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Erreur modèles : {str(e)}",
                "details": {}
            }

    async def check_dependencies(self) -> Dict[str, Any]:
        """Vérifie toutes les dépendances."""
        checks = await asyncio.gather(
            self.check_database(),
            self.check_mlflow(),
            self.check_redis(),
            self.check_models(),
            return_exceptions=True
        )

        database_check, mlflow_check, redis_check, models_check = checks

        return {
            "database": database_check,
            "mlflow": mlflow_check,
            "redis": redis_check,
            "models": models_check
        }

    async def comprehensive_check(self) -> Dict[str, Any]:
        """Check de santé complet."""
        uptime = time.time() - self.start_time
        dependencies = await self.check_dependencies()

        # Déterminer le statut global
        all_statuses = [
            check.get("status", "unknown")
            for check in dependencies.values()
            if isinstance(check, dict)
        ]

        if all(status == "healthy" for status in all_statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in all_statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"

        return {
            "status": overall_status,
            "timestamp": time.time(),
            "uptime_seconds": round(uptime, 2),
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "dependencies": dependencies
        }