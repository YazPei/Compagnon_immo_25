import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict

import psutil
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class MonitoringService:
    """Service de monitoring avec Prometheus."""

    def __init__(self):
        # Configuration
        self.enabled = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"

        if not self.enabled:
            logger.warning("⚠️ Monitoring désactivé par configuration")
            return

        # Initialiser le registry
        self.registry = CollectorRegistry()

        # Métriques HTTP
        self.http_requests_total = Counter(
            "http_requests_total",
            "Nombre total de requêtes HTTP",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.http_request_duration = Histogram(
            "http_request_duration_seconds",
            "Durée des requêtes HTTP en secondes",
            ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )

        # Métriques ML
        self.ml_prediction_count = Counter(
            "ml_prediction_count",
            "Nombre de prédictions effectuées",
            ["model_name", "status"],
            registry=self.registry,
        )

        self.ml_prediction_duration = Histogram(
            "ml_prediction_duration_seconds",
            "Durée des prédictions en secondes",
            ["model_name"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 2.5, 5),
            registry=self.registry,
        )

        self.ml_model_loaded = Gauge(
            "ml_model_loaded",
            "Indique si le modèle est chargé (1) ou non (0)",
            ["model_name"],
            registry=self.registry,
        )

        # Métriques DVC
        self.dvc_sync_count = Counter(
            "dvc_sync_count",
            "Nombre de synchronisations DVC",
            ["status"],
            registry=self.registry,
        )

        self.dvc_sync_duration = Histogram(
            "dvc_sync_duration_seconds",
            "Durée des synchronisations DVC en secondes",
            buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 120),
            registry=self.registry,
        )

        # Métriques système
        self.system_memory_usage = Gauge(
            "system_memory_usage_percent",
            "Utilisation mémoire en pourcentage",
            registry=self.registry,
        )

        self.system_cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "Utilisation CPU en pourcentage",
            registry=self.registry,
        )

        self.system_disk_usage = Gauge(
            "system_disk_usage_percent",
            "Utilisation disque en pourcentage",
            registry=self.registry,
        )

        # Démarrer la collecte en arrière-plan
        self._start_system_metrics_collection()

        logger.info("✅ Service de monitoring initialisé")

    def track_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Suivre une requête HTTP."""
        if not self.enabled:
            return

        self.http_requests_total.labels(
            method=method, endpoint=endpoint, status=str(status_code)
        ).inc()

        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(
            duration
        )

    def track_prediction(
        self, model_name: str, duration: float, status: str = "success"
    ):
        """Suivre une prédiction ML."""
        if not self.enabled:
            return

        self.ml_prediction_count.labels(model_name=model_name, status=status).inc()

        self.ml_prediction_duration.labels(model_name=model_name).observe(duration)

    def update_model_status(self, model_name: str, is_loaded: bool):
        """Mettre à jour le statut d'un modèle."""
        if not self.enabled:
            return

        self.ml_model_loaded.labels(model_name=model_name).set(1 if is_loaded else 0)

    def track_dvc_sync(self, duration: float, status: str = "success"):
        """Suivre une synchronisation DVC."""
        if not self.enabled:
            return

        self.dvc_sync_count.labels(status=status).inc()

        self.dvc_sync_duration.observe(duration)

    def _update_system_metrics(self):
        """Mettre à jour les métriques système."""
        if not self.enabled:
            return

        try:
            # Mémoire
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)

            # CPU
            self.system_cpu_usage.set(psutil.cpu_percent(interval=1))

            # Disque
            disk = psutil.disk_usage("/")
            self.system_disk_usage.set(disk.percent)
        except Exception as e:
            logger.error(
                f"❌ Erreur lors de la mise à jour des métriques système : {e}"
            )

    def _start_system_metrics_collection(self):
        """Démarrer la collecte des métriques système en arrière-plan."""
        if not self.enabled:
            return

        def collect_metrics():
            while True:
                try:
                    self._update_system_metrics()
                except Exception as e:
                    logger.error(f"❌ Erreur collecte métriques : {e}")
                time.sleep(15)  # Toutes les 15 secondes

        # Démarrer dans un thread
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()


# Instance globale
monitoring_service = MonitoringService()
