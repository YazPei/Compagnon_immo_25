import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import logging
from typing import Callable
from app.api.config.settings import settings

logger = logging.getLogger(__name__)

# Métriques Prometheus
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Active HTTP requests'
)

# Métriques système
CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
MEMORY_AVAILABLE = Gauge('system_memory_available_bytes', 'System memory available in bytes')
DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')

# Métriques ML
ML_PREDICTIONS = Counter('ml_predictions_total', 'Total ML predictions', ['model_name', 'status'])
ML_INFERENCE_TIME = Histogram('ml_inference_duration_seconds', 'ML inference duration', ['model_name'])
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy', ['model_name'])

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware pour collecter des métriques de monitoring."""
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/metrics", "/health"]
        
        # Démarrer la collecte des métriques système
        self._start_system_metrics_collection()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request et collecter les métriques."""
        
        # Skip monitoring pour certains endpoints
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Incrémenter les requêtes actives
        ACTIVE_REQUESTS.inc()
        
        # Démarrer le timer
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculer la durée
            duration = time.time() - start_time
            
            # Extraire l'endpoint (sans paramètres dynamiques)
            endpoint = self._get_endpoint_name(request)
            
            # Enregistrer les métriques
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            return response
            
        except Exception as e:
            # Enregistrer l'erreur
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=self._get_endpoint_name(request),
                status_code=500
            ).inc()
            
            logger.error(f"Error in monitoring middleware: {e}")
            raise
        finally:
            # Décrémenter les requêtes actives
            ACTIVE_REQUESTS.dec()
    
    def _get_endpoint_name(self, request: Request) -> str:
        """Extraire le nom de l'endpoint sans paramètres dynamiques."""
        path = request.url.path
        
        # Remplacer les IDs par des placeholders
        import re
        path = re.sub(r'/\d+', '/{id}', path)
        path = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', path)  # UUIDs
        
        return path
    
    def _start_system_metrics_collection(self):
        """Démarrer la collecte des métriques système."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)
            
            # Mémoire
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            MEMORY_AVAILABLE.set(memory.available)
            
            # Disque
            disk = psutil.disk_usage('/')
            DISK_USAGE.set(disk.percent)
            
        except Exception as e:
            logger.warning(f"Could not collect system metrics: {e}")

class MLMetrics:
    """Classe pour enregistrer les métriques ML."""
    
    @staticmethod
    def record_prediction(model_name: str, inference_time: float, success: bool = True):
        """Enregistrer une prédiction ML."""
        status = "success" if success else "error"
        ML_PREDICTIONS.labels(model_name=model_name, status=status).inc()
        ML_INFERENCE_TIME.labels(model_name=model_name).observe(inference_time)
    
    @staticmethod
    def update_model_accuracy(model_name: str, accuracy: float):
        """Mettre à jour la précision du modèle."""
        MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)

def get_metrics():
    """Retourner les métriques Prometheus."""
    return generate_latest()

def get_health_check() -> dict:
    """Retourner le status de santé de l'application."""
    try:
        # Vérifications basiques
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Déterminer le status global
        status = "healthy"
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "degraded"
        
        return {
            "status": status,
            "timestamp": time.time(),
            "version": settings.APP_VERSION,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "database": "connected",  # À implémenter avec check_db_connection()
            "mlflow": "connected"     # À implémenter
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }