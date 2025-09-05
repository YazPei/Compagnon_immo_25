"""
Middleware de sécurité pour l'API
"""
import time
import logging
import asyncio
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware

from app.api.config.settings import settings

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware pour ajouter les headers de sécurité"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Headers de sécurité
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=self"
        }
        
        # HSTS seulement en HTTPS
        if request.url.scheme == "https" or settings.ENVIRONMENT == "production":
            security_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Appliquer les headers
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response

class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Middleware pour limiter la taille des requêtes"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        
        if content_length:
            try:
                content_length = int(content_length)
                if content_length > settings.MAX_REQUEST_SIZE:
                    logger.warning(f"Request too large: {content_length} bytes from {request.client.host}")
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request too large",
                            "max_size": settings.MAX_REQUEST_SIZE,
                            "received_size": content_length
                        }
                    )
            except ValueError:
                logger.warning(f"Invalid content-length header: {content_length}")
        
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware de timeout"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await asyncio.wait_for(
                call_next(request), 
                timeout=settings.REQUEST_TIMEOUT
            )
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {request.url}")
            return JSONResponse(
                status_code=408,
                content={
                    "error": "Request timeout",
                    "timeout": settings.REQUEST_TIMEOUT
                }
            )


class APIVersionMiddleware(BaseHTTPMiddleware):
    """Middleware pour ajouter des informations sur la version de l'API"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        response.headers["X-API-Version"] = settings.APP_VERSION
        response.headers["X-Service-Name"] = settings.PROJECT_NAME
        
        return response


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware CORS avec vérifications de sécurité"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("origin")
        
        # En production, vérifier strictement les origines
        if settings.ENVIRONMENT == "production" and origin:
            allowed_origins = settings.CORS_ORIGINS
            if origin not in allowed_origins:
                logger.warning(f"CORS: Origin non autorisée: {origin}")
                return JSONResponse(
                    status_code=403,
                    content={"error": "Origin not allowed"}
                )
        
        return await call_next(request)


# Fonctions utilitaires pour les middlewares individuels (legacy support)
async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Version fonction du middleware de sécurité (deprecated)"""
    middleware = SecurityHeadersMiddleware(app=None)
    return await middleware.dispatch(request, call_next)


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """Logger les requêtes et réponses"""
    start_time = time.time()
    
    # Informations sur la requête
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "Unknown")
    
    # Log de la requête entrante
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {client_ip} ({user_agent[:50]}...)"
    )
    
    try:
        # Traiter la requête
        response = await call_next(request)
        
        # Calculer le temps de traitement
        process_time = time.time() - start_time
        
        # Log de la réponse
        log_level = logging.ERROR if response.status_code >= 400 else logging.INFO
        logger.log(
            log_level,
            f"Response: {response.status_code} "
            f"in {process_time:.2f}s for {request.method} {request.url.path}"
        )
        
        # Ajouter le temps de traitement aux headers
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Error: {str(e)} in {process_time:.2f}s "
            f"for {request.method} {request.url.path}"
        )
        raise


async def cors_security_middleware(request: Request, call_next: Callable) -> Response:
    """Version fonction du middleware CORS (deprecated)"""
    middleware = CORSSecurityMiddleware(app=None)
    return await middleware.dispatch(request, call_next)


async def api_version_middleware(request: Request, call_next: Callable) -> Response:
    """Version fonction du middleware de version (deprecated)"""
    middleware = APIVersionMiddleware(app=None)
    return await middleware.dispatch(request, call_next)


async def request_size_middleware(request: Request, call_next: Callable) -> Response:
    """Version fonction du middleware de taille (deprecated)"""
    middleware = RequestSizeMiddleware(app=None)
    return await middleware.dispatch(request, call_next)


async def timeout_middleware(request: Request, call_next: Callable) -> Response:
    """Version fonction du middleware de timeout (deprecated)"""
    middleware = TimeoutMiddleware(app=None)
    return await middleware.dispatch(request, call_next)