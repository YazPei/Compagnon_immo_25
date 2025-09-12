"""
Middleware de sécurité pour l'API Compagnon Immobilier
"""
import time
import logging
import asyncio
from typing import Callable, List
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Import conditionnel des settings
try:
    from app.api.config.settings import settings
except ImportError:
    # Fallback si settings n'existe pas
    import os
    class Settings:
        ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
        PROJECT_NAME = "Compagnon Immobilier"
        REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
        CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    
    settings = Settings()

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware pour ajouter les headers de sécurité."""
    
    def __init__(self, app, additional_headers: dict = None):
        super().__init__(app)
        self.additional_headers = additional_headers or {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Ajouter les headers de sécurité à toutes les réponses."""
        response = await call_next(request)
        
        # Headers de sécurité standards
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "X-Permitted-Cross-Domain-Policies": "none"
        }
        
        # HSTS seulement en HTTPS ou production
        if request.url.scheme == "https" or getattr(settings, 'ENVIRONMENT', '') == "production":
            security_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Headers spécifiques à l'environnement
        if getattr(settings, 'ENVIRONMENT', '') == "development":
            # Plus permissif en développement
            security_headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval'; connect-src 'self' ws: wss:;"
        
        # Appliquer les headers de sécurité
        for header, value in security_headers.items():
            response.headers[header] = value
        
        # Ajouter les headers additionnels
        for header, value in self.additional_headers.items():
            response.headers[header] = value
        
        # Headers informatifs sur l'API
        response.headers["X-API-Version"] = getattr(settings, 'APP_VERSION', '1.0.0')
        response.headers["X-Service-Name"] = getattr(settings, 'PROJECT_NAME', 'Compagnon Immobilier')
        
        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Middleware pour limiter la taille des requêtes."""
    
    def __init__(self, app, max_size: int = None):
        super().__init__(app)
        self.max_size = max_size or getattr(settings, 'MAX_REQUEST_SIZE', 10 * 1024 * 1024)  # 10MB par défaut
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Vérifier la taille de la requête."""
        content_length = request.headers.get("content-length")
        
        if content_length:
            try:
                content_length = int(content_length)
                if content_length > self.max_size:
                    client_ip = self._get_client_ip(request)
                    logger.warning(
                        f"🚨 Request too large: {content_length:,} bytes "
                        f"(max: {self.max_size:,}) from {client_ip} "
                        f"for {request.method} {request.url.path}"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Payload Too Large",
                            "message": f"Request size {content_length:,} bytes exceeds maximum {self.max_size:,} bytes",
                            "max_size_mb": round(self.max_size / (1024 * 1024), 1),
                            "received_size_mb": round(content_length / (1024 * 1024), 1)
                        }
                    )
            except ValueError:
                logger.warning(f"Invalid content-length header: {content_length}")
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP du client."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware de timeout pour les requêtes."""
    
    def __init__(self, app, timeout: int = None):
        super().__init__(app)
        self.timeout = timeout or getattr(settings, 'REQUEST_TIMEOUT', 30)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Appliquer un timeout sur les requêtes."""
        try:
            response = await asyncio.wait_for(
                call_next(request), 
                timeout=self.timeout
            )
            return response
            
        except asyncio.TimeoutError:
            client_ip = self._get_client_ip(request)
            logger.error(
                f"⏰ Request timeout after {self.timeout}s - "
                f"{request.method} {request.url.path} from {client_ip}"
            )
            return JSONResponse(
                status_code=408,
                content={
                    "error": "Request Timeout",
                    "message": f"Request exceeded {self.timeout} seconds timeout",
                    "timeout_seconds": self.timeout,
                    "endpoint": request.url.path
                }
            )
        except Exception as e:
            logger.error(f"Unexpected error in timeout middleware: {e}")
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP du client."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


class APIVersionMiddleware(BaseHTTPMiddleware):
    """Middleware pour gérer les versions de l'API."""
    
    def __init__(self, app, current_version: str = None, deprecated_versions: List[str] = None):
        super().__init__(app)
        self.current_version = current_version or getattr(settings, 'APP_VERSION', '1.0.0')
        self.deprecated_versions = deprecated_versions or []
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Gérer la version de l'API."""
        # Récupérer la version demandée
        requested_version = request.headers.get("X-API-Version") or request.query_params.get("version")
        
        # Vérifier si la version est dépréciée
        if requested_version and requested_version in self.deprecated_versions:
            logger.warning(
                f"⚠️ Deprecated API version {requested_version} used "
                f"for {request.method} {request.url.path}"
            )
        
        response = await call_next(request)
        
        # Ajouter les headers de version
        response.headers["X-API-Version"] = self.current_version
        response.headers["X-Service-Name"] = getattr(settings, 'PROJECT_NAME', 'Compagnon Immobilier')
        
        # Ajouter un warning si version dépréciée
        if requested_version and requested_version in self.deprecated_versions:
            response.headers["Warning"] = f"299 - \"API version {requested_version} is deprecated\""
        
        return response


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware CORS avec vérifications de sécurité renforcées."""
    
    def __init__(self, app, allowed_origins: List[str] = None, strict_mode: bool = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or getattr(settings, 'CORS_ORIGINS', [])
        self.strict_mode = strict_mode or (getattr(settings, 'ENVIRONMENT', '') == "production")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Vérifications CORS avec sécurité renforcée."""
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        
        # En mode strict (production), vérifier strictement les origines
        if self.strict_mode and origin:
            if not self._is_origin_allowed(origin):
                client_ip = self._get_client_ip(request)
                logger.warning(
                    f"🚫 CORS: Origin non autorisée '{origin}' "
                    f"from {client_ip} for {request.method} {request.url.path}"
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Forbidden",
                        "message": "Origin not allowed",
                        "allowed_origins": self.allowed_origins if not self.strict_mode else ["Contact administrator"]
                    }
                )
        
        # Vérifications supplémentaires pour les requêtes sensibles
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            if not self._validate_request_headers(request):
                logger.warning(
                    f"🚫 Suspicious request headers detected "
                    f"for {request.method} {request.url.path}"
                )
        
        response = await call_next(request)
        
        # Ajouter headers CORS si nécessaire
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Vérifie si l'origine est autorisée."""
        if not self.strict_mode:
            return True
        
        # Permettre localhost en développement
        if not self.strict_mode and ("localhost" in origin or "127.0.0.1" in origin):
            return True
        
        return origin in self.allowed_origins
    
    def _validate_request_headers(self, request: Request) -> bool:
        """Validation basique des headers de requête."""
        # Vérifier la présence d'un User-Agent valide
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) < 10:  # User-Agent trop court, potentiellement suspect
            return False
        
        # Autres vérifications peuvent être ajoutées ici
        return True
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP du client."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# Fonctions middleware simplifiées pour compatibilité
async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Fonction middleware simplifiée pour headers de sécurité."""
    middleware = SecurityHeadersMiddleware(None)
    return await middleware.dispatch(request, call_next)


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """Logger les requêtes et réponses - version simplifiée."""
    start_time = time.time()
    
    # Informations sur la requête
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()
    
    user_agent = request.headers.get("user-agent", "Unknown")
    
    # Log de la requête entrante
    logger.info(
        f"📥 Request: {request.method} {request.url.path} "
        f"from {client_ip} - UA: {user_agent[:50]}"
    )
    
    # Traiter la requête
    response = await call_next(request)
    
    # Calculer le temps de traitement
    process_time = time.time() - start_time
    
    # Log de la réponse avec status approprié
    status_emoji = "✅" if response.status_code < 400 else "⚠️" if response.status_code < 500 else "🚨"
    logger.info(
        f"{status_emoji} Response: {response.status_code} "
        f"in {process_time:.3f}s for {request.method} {request.url.path}"
    )
    
    # Ajouter le temps de traitement aux headers
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    
    return response


async def cors_security_middleware(request: Request, call_next: Callable) -> Response:
    """Fonction middleware CORS simplifiée."""
    middleware = CORSSecurityMiddleware(None)
    return await middleware.dispatch(request, call_next)


async def api_version_middleware(request: Request, call_next: Callable) -> Response:
    """Fonction middleware version API simplifiée."""
    middleware = APIVersionMiddleware(None)
    return await middleware.dispatch(request, call_next)


async def request_size_middleware(request: Request, call_next: Callable) -> Response:
    """Fonction middleware taille requête simplifiée."""
    middleware = RequestSizeMiddleware(None)
    return await middleware.dispatch(request, call_next)


async def timeout_middleware(request: Request, call_next: Callable) -> Response:
    """Fonction middleware timeout simplifiée."""
    middleware = TimeoutMiddleware(None)
    return await middleware.dispatch(request, call_next)