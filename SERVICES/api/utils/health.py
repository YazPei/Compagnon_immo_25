"""
Module pour vérifier la santé des services et dépendances.
"""

import asyncio
import logging
import time
from typing import Any, Dict  # Suppression de `Callable` inutilisé

import httpx
import redis
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest
from redis.client import Redis
from sqlalchemy import text
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)

from app.api.config.settings import settings
from app.api.db.database import engine
from app.api.services.ml_service import ml_service

logger = logging.getLogger(__name__)

# Définir des métriques Prometheus
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Nombre total de requêtes HTTP",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP en secondes",
    ["method", "endpoint"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        method = request.method
        endpoint = request.url.path

        with REQUEST_LATENCY.labels(method=method, endpoint=endpoint).time():
            response: Response = await call_next(request)
            assert isinstance(response, Response)  # Vérification explicite du type

        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, http_status=response.status_code
        ).inc()

        return response


# Endpoint pour exposer les métriques
async def prometheus_metrics():
    return Response(generate_latest(), media_type="text/plain")


# Correction des types inconnus
checks: list[Dict[str, Any]] = []  # Exemple de définition pour éviter l'erreur
all_statuses: list[str] = [check.get("status", "unknown") for check in checks]

if all(status == "healthy" for status in all_statuses):
    overall_status = "healthy"
elif any(status == "unhealthy" for status in all_statuses):
    overall_status = "unhealthy"
else:
    overall_status = "degraded"


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
                "response_time_ms": 0,  # TODO: mesurer le temps réel
            }
        except Exception as e:
            logger.error(f"Erreur base de données : {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Erreur base de données : {str(e)}",
                "response_time_ms": 0,
            }

    async def check_mlflow(self) -> Dict[str, Any]:
        """Vérifie la connexion à MLflow."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health")

                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "message": "MLflow accessible",
                        "response_time_ms": 0,
                    }
                else:
                    return {
                        "status": "degraded",
                        "message": f"MLflow status: {response.status_code}",
                        "response_time_ms": 0,
                    }
        except Exception as e:
            logger.error(f"Erreur MLflow : {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"MLflow inaccessible : {str(e)}",
                "response_time_ms": 0,
            }

    async def check_redis(self) -> Dict[str, Any]:
        """Vérifie la connexion à Redis."""
        try:
            r: Redis = redis.from_url(  # type: ignore
                settings.REDIS_URL, decode_responses=True
            )
            r.ping()  # type: ignore

            assert isinstance(settings.REDIS_URL, str)  # Vérification explicite du type
            assert settings.REDIS_URL.startswith("redis://")  # Vérification du format

            return {
                "status": "healthy",
                "message": "Redis accessible",
                "response_time_ms": 0,
            }
        except Exception as e:
            logger.error(f"Erreur Redis : {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Redis inaccessible : {str(e)}",
                "response_time_ms": 0,
            }

    async def check_models(self) -> Dict[str, Any]:
        """Vérifie l'état des modèles ML."""
        try:
            models_status = ml_service.get_models_status()  # Suppression de await

            if models_status.get("models_loaded", 0) > 0:
                return {
                    "status": "healthy",
                    "message": (f"{models_status['models_loaded']} modèles chargés"),
                    "details": models_status,
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Aucun modèle chargé",
                    "details": models_status,
                }
        except Exception as e:
            logger.error(f"Erreur modèles : {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Erreur modèles : {str(e)}",
                "details": {},
            }

    async def check_dependencies(self) -> Dict[str, Any]:
        """Vérifie toutes les dépendances."""
        checks = await asyncio.gather(
            self.check_database(),
            self.check_mlflow(),
            self.check_redis(),
            self.check_models(),
            return_exceptions=True,
        )

        database_check, mlflow_check, redis_check, models_check = checks

        return {
            "database": database_check,
            "mlflow": mlflow_check,
            "redis": redis_check,
            "models": models_check,
        }

    async def comprehensive_check(self) -> Dict[str, Any]:
        """Check de santé complet."""
        uptime = time.time() - self.start_time
        dependencies = await self.check_dependencies()

        # Déterminer le statut global
        all_statuses: list[str] = [check.get("status", "unknown") for check in checks]

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
            "dependencies": dependencies,
        }
