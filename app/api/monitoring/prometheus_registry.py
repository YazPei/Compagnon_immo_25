"""
Registry centralisé pour toutes les métriques Prometheus.
"""

import time
from typing import Any, Dict

from prometheus_client import (CollectorRegistry, Counter, Gauge, Histogram,
                               Info)

# Registry centralisé pour toutes les métriques
PROMETHEUS_REGISTRY = CollectorRegistry()

# Métriques HTTP
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total des requêtes HTTP par méthode et endpoint",
    ["method", "endpoint", "status_code"],
    registry=PROMETHEUS_REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP en secondes",
    ["method", "endpoint"],
    registry=PROMETHEUS_REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

HTTP_REQUEST_SIZE = Histogram(
    "http_request_size_bytes",
    "Taille des requêtes HTTP en bytes",
    ["method", "endpoint"],
    registry=PROMETHEUS_REGISTRY,
)

HTTP_RESPONSE_SIZE = Histogram(
    "http_response_size_bytes",
    "Taille des réponses HTTP en bytes",
    ["method", "endpoint"],
    registry=PROMETHEUS_REGISTRY,
)

# Métriques de l'application
API_INFO = Info("api_info", "Informations sur l'API", registry=PROMETHEUS_REGISTRY)

ACTIVE_CONNECTIONS = Gauge(
    "active_connections", "Nombre de connexions actives", registry=PROMETHEUS_REGISTRY
)

# Métriques ML et modèles
MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Nombre total de prédictions par modèle",
    ["model_name", "model_version"],
    registry=PROMETHEUS_REGISTRY,
)

MODEL_PREDICTION_DURATION = Histogram(
    "model_prediction_duration_seconds",
    "Durée des prédictions par modèle",
    ["model_name", "model_version"],
    registry=PROMETHEUS_REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODEL_ERRORS_TOTAL = Counter(
    "model_errors_total",
    "Nombre total d'erreurs par modèle",
    ["model_name", "error_type"],
    registry=PROMETHEUS_REGISTRY,
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Temps de chargement des modèles",
    ["model_name", "model_version"],
    registry=PROMETHEUS_REGISTRY,
)

MODELS_LOADED = Gauge(
    "models_loaded_total", "Nombre de modèles chargés", registry=PROMETHEUS_REGISTRY
)

# Métriques des dépendances
DEPENDENCY_STATUS = Gauge(
    "dependency_status",
    "Statut des dépendances (1=healthy, 0=unhealthy)",
    ["dependency_name"],
    registry=PROMETHEUS_REGISTRY,
)

DEPENDENCY_RESPONSE_TIME = Histogram(
    "dependency_response_time_seconds",
    "Temps de réponse des dépendances",
    ["dependency_name"],
    registry=PROMETHEUS_REGISTRY,
)

DATABASE_CONNECTIONS_ACTIVE = Gauge(
    "database_connections_active",
    "Nombre de connexions actives à la base de données",
    registry=PROMETHEUS_REGISTRY,
)

REDIS_OPERATIONS_TOTAL = Counter(
    "redis_operations_total",
    "Nombre total d'opérations Redis",
    ["operation_type", "status"],
    registry=PROMETHEUS_REGISTRY,
)

# Métriques business
ESTIMATION_REQUESTS_TOTAL = Counter(
    "estimation_requests_total",
    "Nombre total de demandes d'estimation",
    ["property_type", "region"],
    registry=PROMETHEUS_REGISTRY,
)

ESTIMATION_VALUE_HISTOGRAM = Histogram(
    "estimation_value_euros",
    "Distribution des valeurs d'estimation",
    ["property_type"],
    registry=PROMETHEUS_REGISTRY,
    buckets=(50000, 100000, 200000, 300000, 500000, 750000, 1000000, 1500000, 2000000),
)

CACHE_OPERATIONS_TOTAL = Counter(
    "cache_operations_total",
    "Nombre total d'opérations de cache",
    ["operation", "status"],
    registry=PROMETHEUS_REGISTRY,
)

CACHE_HIT_RATE = Gauge(
    "cache_hit_rate", "Taux de succès du cache (0-1)", registry=PROMETHEUS_REGISTRY
)

# Métriques système
MEMORY_USAGE_BYTES = Gauge(
    "memory_usage_bytes", "Utilisation mémoire en bytes", registry=PROMETHEUS_REGISTRY
)

CPU_USAGE_PERCENT = Gauge(
    "cpu_usage_percent", "Utilisation CPU en pourcentage", registry=PROMETHEUS_REGISTRY
)


class MetricsCollector:
    """Collecteur centralisé pour les métriques."""

    def __init__(self):
        self.start_time = time.time()

    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
    ):
        """Enregistre une requête HTTP."""
        HTTP_REQUESTS_TOTAL.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

        HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

        if request_size > 0:
            HTTP_REQUEST_SIZE.labels(method=method, endpoint=endpoint).observe(
                request_size
            )

        if response_size > 0:
            HTTP_RESPONSE_SIZE.labels(method=method, endpoint=endpoint).observe(
                response_size
            )

    def record_model_prediction(
        self, model_name: str, model_version: str, duration: float, success: bool = True
    ):
        """Enregistre une prédiction de modèle."""
        MODEL_PREDICTIONS_TOTAL.labels(
            model_name=model_name, model_version=model_version
        ).inc()

        MODEL_PREDICTION_DURATION.labels(
            model_name=model_name, model_version=model_version
        ).observe(duration)

        if not success:
            MODEL_ERRORS_TOTAL.labels(
                model_name=model_name, error_type="prediction_failed"
            ).inc()

    def record_estimation(self, property_type: str, region: str, value: float):
        """Enregistre une estimation immobilière."""
        ESTIMATION_REQUESTS_TOTAL.labels(
            property_type=property_type, region=region
        ).inc()

        ESTIMATION_VALUE_HISTOGRAM.labels(property_type=property_type).observe(value)

    def update_dependency_status(
        self, dependency_name: str, is_healthy: bool, response_time: float = 0
    ):
        """Met à jour le statut d'une dépendance."""
        DEPENDENCY_STATUS.labels(dependency_name=dependency_name).set(
            1 if is_healthy else 0
        )

        if response_time > 0:
            DEPENDENCY_RESPONSE_TIME.labels(dependency_name=dependency_name).observe(
                response_time
            )

    def record_cache_operation(self, operation: str, hit: bool):
        """Enregistre une opération de cache."""
        status = "hit" if hit else "miss"
        CACHE_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()

    def update_system_metrics(self, memory_usage: int, cpu_usage: float):
        """Met à jour les métriques système."""
        MEMORY_USAGE_BYTES.set(memory_usage)
        CPU_USAGE_PERCENT.set(cpu_usage)


# Instance globale du collecteur
metrics_collector = MetricsCollector()

# Initialisation des métriques d'information
API_INFO.info({"version": "1.0.0", "environment": "production"})
