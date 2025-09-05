from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import json
import uuid
from typing import Callable
from app.api.config.settings import settings

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour logger toutes les requêtes avec tracing."""
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        
        # Skip logging pour certains endpoints
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Générer un ID de trace unique
        trace_id = str(uuid.uuid4())[:8]
        request.state.trace_id = trace_id
        
        # Début de la requête
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log de la requête entrante
        logger.info(
            f"📥 [{trace_id}] {request.method} {request.url.path} - "
            f"IP: {client_ip} - UA: {user_agent[:50]}"
        )
        
        # Traitement de la requête
        try:
            response = await call_next(request)
            
            # Calcul du temps de traitement
            process_time = time.time() - start_time
            
            # Log de la réponse
            log_data = {
                "trace_id": trace_id,
                "method": request.method,
                "url": str(request.url.path),
                "query_params": dict(request.query_params) if request.query_params else {},
                "status_code": response.status_code,
                "process_time": round(process_time, 4),
                "client_ip": client_ip,
                "user_agent": user_agent[:100],
                "content_length": response.headers.get("content-length", "unknown")
            }
            
            # Log selon le niveau de status code
            if response.status_code >= 500:
                logger.error(f"🚨 [{trace_id}] Server Error - {json.dumps(log_data)}")
            elif response.status_code >= 400:
                logger.warning(f"⚠️ [{trace_id}] Client Error - {json.dumps(log_data)}")
            else:
                logger.info(f"✅ [{trace_id}] Success - {process_time:.4f}s - {request.method} {request.url.path}")
            
            # Ajouter headers de trace et temps
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Trace-ID"] = trace_id
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"❌ [{trace_id}] Exception in request {request.method} {request.url.path} - "
                f"Time: {process_time:.4f}s - Error: {str(e)}",
                exc_info=True
            )
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP réelle du client (même derrière un proxy)."""
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"