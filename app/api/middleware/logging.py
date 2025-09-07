from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import json
import uuid
from typing import Callable
import os
from datetime import datetime

# Import conditionnel des settings
try:
    from app.api.config.settings import settings
except ImportError:
    # Fallback si settings n'existe pas
    class Settings:
        DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    settings = Settings()

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour logger toutes les requêtes avec tracing."""
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health", "/metrics", "/docs", "/openapi.json", 
            "/favicon.ico", "/static"
        ]
    
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
        
        # Log de la requête entrante avec plus de détails
        request_size = request.headers.get("content-length", "0")
        
        logger.info(
            f"📥 [{trace_id}] {request.method} {request.url.path} - "
            f"IP: {client_ip} - Size: {request_size}B - "
            f"UA: {user_agent[:50]}"
        )
        
        # Traitement de la requête
        try:
            response = await call_next(request)
            
            # Calcul du temps de traitement
            process_time = time.time() - start_time
            
            # Log de la réponse avec métriques détaillées
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": trace_id,
                "method": request.method,
                "url": str(request.url.path),
                "query_params": dict(request.query_params) if request.query_params else {},
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "client_ip": client_ip,
                "user_agent": user_agent[:100],
                "request_size": request_size,
                "response_size": response.headers.get("content-length", "unknown"),
                "environment": getattr(settings, 'ENVIRONMENT', 'unknown')
            }
            
            # Ajout d'informations spécifiques aux endpoints d'estimation
            if "/estimation" in request.url.path:
                log_data["endpoint_type"] = "estimation"
                log_data["business_critical"] = True
            elif "/historique" in request.url.path:
                log_data["endpoint_type"] = "historique"
            elif "/dvc" in request.url.path or "/ml" in request.url.path:
                log_data["endpoint_type"] = "mlops"
                log_data["business_critical"] = True
            
            # Log selon le niveau de status code avec métriques
            if response.status_code >= 500:
                logger.error(
                    f"🚨 [{trace_id}] Server Error - "
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s - "
                    f"Details: {json.dumps(log_data, indent=None)}"
                )
            elif response.status_code >= 400:
                logger.warning(
                    f"⚠️ [{trace_id}] Client Error - "
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s - "
                    f"IP: {client_ip}"
                )
            else:
                # Log succinct pour les succès
                logger.info(
                    f"✅ [{trace_id}] {request.method} {request.url.path} - "
                    f"{response.status_code} - {process_time:.4f}s"
                )
                
                # Log détaillé seulement en mode DEBUG
                if getattr(settings, 'DEBUG', False):
                    logger.debug(f"Debug details: {json.dumps(log_data, indent=2)}")
            
            # Ajouter headers de trace et métriques
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Environment"] = getattr(settings, 'ENVIRONMENT', 'unknown')
            
            # Métriques de performance pour les endpoints critiques
            if process_time > 5.0:  # Plus de 5 secondes
                logger.warning(
                    f"⏰ [{trace_id}] Slow request detected - "
                    f"{request.method} {request.url.path} - "
                    f"Time: {process_time:.4f}s"
                )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            error_data = {
                "trace_id": trace_id,
                "method": request.method,
                "url": str(request.url.path),
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time": round(process_time, 4),
                "client_ip": client_ip,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.error(
                f"❌ [{trace_id}] Exception in {request.method} {request.url.path} - "
                f"Time: {process_time:.4f}s - Error: {str(e)} - "
                f"Details: {json.dumps(error_data)}"
            )
            
            # Log stack trace seulement en mode DEBUG
            if getattr(settings, 'DEBUG', False):
                logger.error(f"Stack trace for [{trace_id}]:", exc_info=True)
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP réelle du client (même derrière un proxy)."""
        # Vérifier les headers de proxy dans l'ordre de priorité
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Prendre la première IP de la liste (client original)
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Headers Cloudflare
        cf_connecting_ip = request.headers.get("CF-Connecting-IP")
        if cf_connecting_ip:
            return cf_connecting_ip.strip()
        
        # Headers AWS ALB
        forwarded = request.headers.get("X-Forwarded")
        if forwarded and "for=" in forwarded:
            try:
                # Extraire l'IP du format X-Forwarded: for=192.168.1.1
                for_part = forwarded.split("for=")[1].split(";")[0]
                return for_part.strip()
            except (IndexError, AttributeError):
                pass
        
        # Fallback sur l'IP de connection directe
        return request.client.host if request.client else "unknown"


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware de logging structuré pour l'analyse et monitoring."""
    
    def __init__(self, app, log_to_file: bool = False, log_file_path: str = None):
        super().__init__(app)
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path or "logs/api_requests.jsonl"
        
        # Créer le dossier de logs si nécessaire
        if self.log_to_file:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log structuré pour analytics."""
        
        start_time = time.time()
        trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4())[:8])
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Structure de log pour analytics
            structured_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": trace_id,
                "request": {
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "headers": {
                        "user_agent": request.headers.get("user-agent", ""),
                        "content_type": request.headers.get("content-type", ""),
                        "api_key": "***" if request.headers.get("x-api-key") else None
                    },
                    "client_ip": self._get_client_ip(request),
                    "size": request.headers.get("content-length", "0")
                },
                "response": {
                    "status_code": response.status_code,
                    "size": response.headers.get("content-length", "0"),
                    "content_type": response.headers.get("content-type", "")
                },
                "metrics": {
                    "process_time_ms": round(process_time * 1000, 2),
                    "is_slow": process_time > 1.0,
                    "is_error": response.status_code >= 400
                },
                "context": {
                    "environment": getattr(settings, 'ENVIRONMENT', 'unknown'),
                    "endpoint_type": self._classify_endpoint(request.url.path)
                }
            }
            
            # Log vers fichier si activé
            if self.log_to_file:
                self._write_to_file(structured_log)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            error_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": trace_id,
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "request_path": request.url.path,
                    "process_time_ms": round(process_time * 1000, 2)
                }
            }
            
            if self.log_to_file:
                self._write_to_file(error_log)
            
            raise
    
    def _classify_endpoint(self, path: str) -> str:
        """Classifie le type d'endpoint pour les métriques."""
        if "/estimation" in path:
            return "estimation"
        elif "/historique" in path:
            return "historique"
        elif "/health" in path:
            return "health"
        elif "/dvc" in path or "/ml" in path:
            return "mlops"
        elif "/deployment" in path:
            return "deployment"
        else:
            return "other"
    
    def _get_client_ip(self, request: Request) -> str:
        """Méthode partagée pour récupérer l'IP client."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _write_to_file(self, log_data: dict):
        """Écrit le log structuré dans un fichier JSONL."""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Erreur écriture log fichier: {e}")


# Fonction utilitaire pour configurer le logging
def setup_logging(level: str = "INFO", format_type: str = "standard"):
    """Configure le système de logging de l'application."""
    
    formats = {
        "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        "json": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    }
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=formats.get(format_type, formats["standard"]),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log", mode="a") if os.path.exists("logs") or os.makedirs("logs", exist_ok=True) else logging.StreamHandler()
        ]
    )
    
    # Réduire le niveau de logging pour certains modules tiers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)