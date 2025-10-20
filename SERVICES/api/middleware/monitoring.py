"""
Middleware de monitoring utilisant le registry centralisé.
"""

import time

import psutil
from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.monitoring.prometheus_registry import (ACTIVE_CONNECTIONS,
                                                    PROMETHEUS_REGISTRY,
                                                    metrics_collector)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware pour collecter les métriques Prometheus."""

    def __init__(self, app, registry=None):
        super().__init__(app)
        self.registry = registry or PROMETHEUS_REGISTRY

    async def dispatch(self, request: Request, call_next):
        """Traite la requête et collecte les métriques."""
        # Ignorer l'endpoint /metrics pour éviter la récursion
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()

        # Incrémenter les connexions actives
        ACTIVE_CONNECTIONS.inc()

        try:
            # Traiter la requête
            response = await call_next(request)

            # Calculer la durée
            duration = time.time() - start_time

            # Collecter les métriques
            metrics_collector.record_http_request(
                method=request.method,
                endpoint=self._get_endpoint_path(request),
                status_code=response.status_code,
                duration=duration,
                request_size=self._get_request_size(request),
                response_size=self._get_response_size(response),
            )

            return response

        except Exception as e:
            # En cas d'erreur, enregistrer une métrique d'erreur
            duration = time.time() - start_time

            metrics_collector.record_http_request(
                method=request.method,
                endpoint=self._get_endpoint_path(request),
                status_code=500,
                duration=duration,
            )

            raise e

        finally:
            # Décrémenter les connexions actives
            ACTIVE_CONNECTIONS.dec()

    def _get_endpoint_path(self, request: Request) -> str:
        """Extrait le chemin de l'endpoint de manière normalisée."""
        path = request.url.path

        # Normaliser les chemins avec des IDs
        if "/api/v1/estimation/" in path:
            return "/api/v1/estimation/{id}"
        elif "/api/v1/models/" in path:
            return "/api/v1/models/{id}"

        return path

    def _get_request_size(self, request: Request) -> int:
        """Calcule la taille de la requête."""
        try:
            return int(request.headers.get("content-length", 0))
        except (ValueError, TypeError):
            return 0

    def _get_response_size(self, response: Response) -> int:
        """Calcule la taille de la réponse."""
        try:
            content_length = response.headers.get("content-length")
            if content_length:
                return int(content_length)

            # Estimer la taille si pas d'en-tête content-length
            if hasattr(response, "body") and response.body:
                return len(response.body)

            return 0
        except (ValueError, TypeError, AttributeError):
            return 0


class SystemMetricsCollector:
    """Collecteur de métriques système."""

    def __init__(self):
        self.process = psutil.Process()

    def collect_system_metrics(self):
        """Collecte les métriques système."""
        try:
            # Métriques mémoire
            memory_info = self.process.memory_info()
            memory_usage = memory_info.rss  # Resident Set Size

            # Métriques CPU
            cpu_percent = self.process.cpu_percent()

            # Mettre à jour les métriques
            metrics_collector.update_system_metrics(
                memory_usage=memory_usage, cpu_usage=cpu_percent
            )

        except Exception as e:
            # Log l'erreur mais ne pas faire échouer l'application
            print(f"Erreur lors de la collecte des métriques système : {e}")


# Instance globale du collecteur système
system_metrics_collector = SystemMetricsCollector()


# Fonction utilitaire pour configurer le middleware
def setup_monitoring_middleware(app, track_system_metrics: bool = True):
    """Configure le middleware de monitoring pour l'application."""
    app.add_middleware(PrometheusMiddleware, track_system_metrics=track_system_metrics)

    logger.info("✅ Monitoring middleware configuré avec métriques Prometheus")
