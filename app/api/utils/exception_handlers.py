"""
Gestionnaires d'exceptions globaux pour l'API.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import time
import traceback
from typing import Union

from app.api.config.settings import settings
from app.api.utils.exceptions import CompagnionImmoException

logger = logging.getLogger(__name__)


async def compagnion_exception_handler(request: Request, exc: CompagnionImmoException):
    """Gestionnaire pour les exceptions personnalisées de l'application."""
    logger.error(f"CompagnionImmoException: {exc.detail} - URL: {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": exc.error_code,
            "detail": exc.detail,
            "type": "application_error",
            "path": str(request.url.path),
            "timestamp": time.time()
        },
        headers=exc.headers
    )


async def http_exception_handler(request: Request, exc: Union[HTTPException, StarletteHTTPException]):
    """Gestionnaire pour les exceptions HTTP standard."""
    logger.error(f"HTTP Exception: {exc.detail} - URL: {request.url} - Status: {exc.status_code}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "detail": exc.detail,
            "type": "http_error",
            "path": str(request.url.path),
            "timestamp": time.time(),
            "status_code": exc.status_code
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Gestionnaire pour les erreurs de validation Pydantic."""
    logger.error(f"Validation error: {exc.errors()} - URL: {request.url}")
    
    # Formatter les erreurs de validation
    formatted_errors = [
        {
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        }
        for error in exc.errors()
    ]
    
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "detail": "Erreurs de validation",
            "type": "validation_error",
            "path": str(request.url.path),
            "timestamp": time.time(),
            "errors": formatted_errors
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Gestionnaire pour toutes les autres exceptions non gérées."""
    # Log complet avec traceback
    logger.error(
        f"Unhandled exception: {str(exc)} - URL: {request.url}",
        exc_info=True
    )
    
    # En mode développement, inclure plus de détails
    if settings.DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "detail": str(exc),
                "type": "internal_error",
                "path": str(request.url.path),
                "timestamp": time.time(),
                "traceback": traceback.format_exc().split('\n')
            }
        )
    else:
        # En mode production, message générique
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "detail": "Une erreur interne est survenue",
                "type": "internal_error",
                "path": str(request.url.path),
                "timestamp": time.time()
            }
        )


async def method_not_allowed_handler(request: Request, exc: Exception):
    """Gestionnaire pour les méthodes non autorisées."""
    logger.warning(f"Method not allowed: {request.method} {request.url}")
    
    return JSONResponse(
        status_code=405,
        content={
            "error": True,
            "detail": f"Méthode {request.method} non autorisée pour {request.url.path}",
            "type": "method_not_allowed",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )


async def not_found_handler(request: Request, exc: Exception):
    """Gestionnaire pour les ressources non trouvées."""
    logger.warning(f"Not found: {request.url}")
    
    return JSONResponse(
        status_code=404,
        content={
            "error": True,
            "detail": f"Ressource non trouvée: {request.url.path}",
            "type": "not_found",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )