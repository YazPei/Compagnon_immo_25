import time
import logging
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from .metrics import http_requests_total, http_request_duration_seconds, metrics

logger = logging.getLogger(__name__)

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware pour capturer les métriques HTTP."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Traitement de la requête
        response = await call_next(request)
        
        # Calcul de la durée
        duration = time.time() - start_time
        
        # Extraction des labels
        method = request.method
        endpoint = self._get_endpoint(request)
        status_code = str(response.status_code)
        
        # Enregistrement des métriques
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Log des requêtes lentes
        if duration > 2.0:
            logger.warning(f"Slow request: {method} {endpoint} took {duration:.2f}s")
        
        # Log des erreurs
        if response.status_code >= 500:
            logger.error(f"Server error: {method} {endpoint} returned {status_code}")
            metrics.record_security_event('server_error', 'high')
        
        # Audit des endpoints sensibles
        if endpoint.startswith('/admin') or endpoint.startswith('/api/v1/admin'):
            logger.info(f"AUDIT: {method} {endpoint} by {self._get_client_info(request)}")
        
        return response
    
    def _get_endpoint(self, request: Request) -> str:
        """Extraire l'endpoint normalisé."""
        path = request.url.path
        
        # Normaliser les IDs dans les URLs
        import re
        path = re.sub(r'/\d+/', '/{id}/', path)
        path = re.sub(r'/\d+$', '/{id}', path)
        
        return path
    
    def _get_client_info(self, request: Request) -> str:
        """Obtenir les informations du client."""
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        else:
            client_ip = request.client.host if request.client else 'unknown'
        
        user_agent = request.headers.get('User-Agent', 'unknown')
        return f"IP:{client_ip} UA:{user_agent[:50]}"

class MLMonitoringMiddleware:
    """Middleware pour monitorer les prédictions ML."""
    
    @staticmethod
    def monitor_prediction(model_type: str, input_data, prediction, confidence: float):
        """Monitorer une prédiction ML."""
        start_time = time.time()
        
        # Calculer la durée (simulée ici)
        duration = time.time() - start_time
        
        # Enregistrer les métriques
        metrics.record_prediction(
            model_type=model_type,
            confidence=confidence,
            duration=duration
        )
        
        # Vérifications de qualité
        if confidence < 0.5:
            logger.warning(f"Low confidence prediction: {confidence:.2f} for {model_type}")
            metrics.record_security_event('low_confidence_prediction', 'medium')
        
        return prediction
