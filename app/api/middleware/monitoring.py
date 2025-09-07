import time
import logging
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import psutil
import os

# Import conditionnel des services
try:
    from app.services.monitoring_service import monitoring_service
except ImportError:
    # Fallback si le service n'existe pas
    class MockMonitoringService:
        def track_request(self, **kwargs):
            pass
        def track_model_prediction(self, **kwargs):
            pass
    monitoring_service = MockMonitoringService()

logger = logging.getLogger(__name__)

# Métriques Prometheus pour le projet Compagnon_immo
REQUEST_COUNT = Counter(
    'compagnon_immo_requests_total', 
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'compagnon_immo_request_duration_seconds', 
    'Request duration in seconds',
    ['method', 'endpoint']
)

MODEL_PREDICTIONS = Counter(
    'compagnon_immo_predictions_total', 
    'Total model predictions',
    ['model_type', 'property_type']
)

ESTIMATION_REQUESTS = Counter(
    'compagnon_immo_estimations_total',
    'Total estimation requests',
    ['status', 'property_type']
)

ACTIVE_CONNECTIONS = Gauge(
    'compagnon_immo_active_connections',
    'Number of active connections'
)

DATABASE_OPERATIONS = Counter(
    'compagnon_immo_database_operations_total',
    'Total database operations',
    ['operation', 'table']
)

MODEL_ACCURACY = Gauge(
    'compagnon_immo_model_accuracy',
    'Current model accuracy score',
    ['model_name']
)

SYSTEM_METRICS = Gauge(
    'compagnon_immo_system_usage',
    'System resource usage',
    ['resource_type']
)

DVC_OPERATIONS = Counter(
    'compagnon_immo_dvc_operations_total',
    'Total DVC operations',
    ['operation', 'status']
)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour le monitoring avancé des requêtes 
    avec métriques spécifiques au projet Compagnon_immo.
    """

    def __init__(self, app, track_system_metrics: bool = True):
        super().__init__(app)
        self.track_system_metrics = track_system_metrics
        self._active_connections = 0

    async def dispatch(self, request: Request, call_next):
        """Process request with comprehensive monitoring."""
        
        # Incrémenter les connexions actives
        self._active_connections += 1
        ACTIVE_CONNECTIONS.set(self._active_connections)
        
        # Temps de début et informations de base
        start_time = time.time()
        method = request.method
        path = request.url.path
        endpoint_category = self._categorize_endpoint(path)
        
        try:
            # Exécuter la requête
            response = await call_next(request)
            
            # Calculer la durée
            duration = time.time() - start_time
            status_code = response.status_code
            
            # Métriques Prometheus de base
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint_category,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint_category
            ).observe(duration)
            
            # Métriques spécifiques aux estimations immobilières
            if "/estimation" in path and status_code == 200:
                property_type = self._extract_property_type(request)
                ESTIMATION_REQUESTS.labels(
                    status="success",
                    property_type=property_type
                ).inc()
                
                # Si c'est une prédiction de modèle ML
                model_type = self._extract_model_type(request, response)
                if model_type:
                    MODEL_PREDICTIONS.labels(
                        model_type=model_type,
                        property_type=property_type
                    ).inc()
            
            elif "/estimation" in path and status_code >= 400:
                property_type = self._extract_property_type(request)
                ESTIMATION_REQUESTS.labels(
                    status="error",
                    property_type=property_type or "unknown"
                ).inc()
            
            # Tracking DVC operations
            if "/dvc" in path:
                operation = self._extract_dvc_operation(path)
                status = "success" if status_code < 400 else "error"
                DVC_OPERATIONS.labels(
                    operation=operation,
                    status=status
                ).inc()
            
            # Enregistrer via le service de monitoring
            monitoring_service.track_request(
                method=method,
                endpoint=path,
                status_code=status_code,
                duration=duration,
                endpoint_category=endpoint_category
            )
            
            # Métriques système périodiques
            if self.track_system_metrics:
                self._update_system_metrics()
            
            # Log des requêtes lentes (> 2s pour les estimations)
            slow_threshold = 2.0 if "/estimation" in path else 5.0
            if duration > slow_threshold:
                logger.warning(
                    f"Slow request detected: {method} {path} "
                    f"took {duration:.2f}s (threshold: {slow_threshold}s)"
                )
            
            return response

        except Exception as e:
            # En cas d'erreur
            duration = time.time() - start_time
            
            # Métriques d'erreur
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint_category,
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint_category
            ).observe(duration)
            
            # Log détaillé de l'erreur
            logger.error(
                f"Error in {method} {path}: {str(e)} "
                f"(duration: {duration:.2f}s, category: {endpoint_category})",
                exc_info=True
            )
            
            # Tracking via service
            monitoring_service.track_request(
                method=method,
                endpoint=path,
                status_code=500,
                duration=duration,
                error=str(e),
                endpoint_category=endpoint_category
            )
            
            # Re-lever l'exception
            raise
        
        finally:
            # Décrémenter les connexions actives
            self._active_connections = max(0, self._active_connections - 1)
            ACTIVE_CONNECTIONS.set(self._active_connections)

    def _categorize_endpoint(self, path: str) -> str:
        """Catégorise l'endpoint pour les métriques."""
        if "/estimation" in path:
            return "estimation"
        elif "/historique" in path:
            return "historique"
        elif "/health" in path:
            return "health"
        elif "/dvc" in path:
            return "dvc"
        elif "/ml" in path:
            return "ml"
        elif "/deployment" in path:
            return "deployment"
        elif "/metrics" in path:
            return "monitoring"
        else:
            return "other"

    def _extract_property_type(self, request: Request) -> Optional[str]:
        """Extrait le type de propriété de la requête."""
        try:
            # Depuis les query params
            if "type_bien" in request.query_params:
                return request.query_params["type_bien"]
            
            # Depuis le body (si disponible)
            if hasattr(request.state, 'body_data'):
                body = request.state.body_data
                return body.get("property_type") or body.get("type_bien")
            
            return "unknown"
        except Exception:
            return "unknown"

    def _extract_model_type(self, request: Request, response: Response) -> Optional[str]:
        """Extrait le type de modèle utilisé."""
        try:
            # Depuis les headers de réponse
            model_header = response.headers.get("X-Model-Used")
            if model_header:
                return model_header
            
            # Depuis les query params de la requête
            model_param = request.query_params.get("model")
            if model_param:
                return model_param
            
            # Modèle par défaut pour les estimations
            if "/estimation" in request.url.path:
                return "lgbm_default"
            
            return None
        except Exception:
            return None

    def _extract_dvc_operation(self, path: str) -> str:
        """Extrait le type d'opération DVC."""
        if "pull" in path:
            return "pull"
        elif "push" in path:
            return "push"
        elif "status" in path:
            return "status"
        elif "pipeline" in path:
            return "pipeline"
        else:
            return "other"

    def _update_system_metrics(self):
        """Met à jour les métriques système."""
        try:
            # CPU
            SYSTEM_METRICS.labels(resource_type="cpu_percent").set(
                psutil.cpu_percent(interval=0.1)
            )
            
            # Memory
            memory = psutil.virtual_memory()
            SYSTEM_METRICS.labels(resource_type="memory_percent").set(
                memory.percent
            )
            
            # Disk
            disk = psutil.disk_usage('/')
            SYSTEM_METRICS.labels(resource_type="disk_percent").set(
                (disk.used / disk.total) * 100
            )
            
            # Nombre de processus
            SYSTEM_METRICS.labels(resource_type="process_count").set(
                len(psutil.pids())
            )
            
        except Exception as e:
            logger.warning(f"Erreur mise à jour métriques système: {e}")


class ModelPerformanceMonitoring:
    """Classe pour le monitoring spécifique des performances des modèles ML."""
    
    @staticmethod
    def track_prediction_accuracy(model_name: str, accuracy: float):
        """Track l'accuracy d'un modèle."""
        MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)
    
    @staticmethod
    def track_database_operation(operation: str, table: str):
        """Track les opérations de base de données."""
        DATABASE_OPERATIONS.labels(
            operation=operation,
            table=table
        ).inc()
    
    @staticmethod
    def track_model_prediction(model_type: str, property_type: str, confidence: float = None):
        """Track une prédiction de modèle avec détails."""
        MODEL_PREDICTIONS.labels(
            model_type=model_type,
            property_type=property_type
        ).inc()
        
        if confidence is not None:
            # Vous pourriez ajouter une métrique de confiance ici
            logger.info(f"Prédiction {model_type} pour {property_type} avec confiance {confidence}")


def get_metrics_endpoint():
    """
    Endpoint FastAPI pour exposer les métriques Prometheus.
    À utiliser dans votre router principal.
    """
    from fastapi import Response
    
    def metrics():
        """Expose les métriques Prometheus au format standard."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    return metrics


# Instance globale pour l'utilisation dans l'application
model_monitor = ModelPerformanceMonitoring()


# Fonction utilitaire pour configurer le middleware
def setup_monitoring_middleware(app, track_system_metrics: bool = True):
    """Configure le middleware de monitoring pour l'application."""
    app.add_middleware(
        MonitoringMiddleware,
        track_system_metrics=track_system_metrics
    )
    
    logger.info("✅ Monitoring middleware configuré avec métriques Prometheus")