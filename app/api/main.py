# app/api/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.config.settings import settings
from app.api.middleware.error_handling import ErrorHandlingMiddleware

from app.api.routes import main as main_routes
from app.api.routes import estimation as estimation_routes
from app.api.routes import health as health_routes
from app.api.routes import historique as historique_routes
from app.api.routes import metrics as metrics_routes

# Importation des gestionnaires d'exceptions personnalisés
from app.api.utils.exception_handlers import (
    http_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from typing import Callable, Awaitable

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API d'estimation immobilière",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ajout du middleware pour la gestion des erreurs
app.add_middleware(ErrorHandlingMiddleware)


# Logging Middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response


# Ajout du middleware pour les logs
app.add_middleware(LoggingMiddleware)


@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
    }


@app.get("/liveness")
async def liveness():
    return {"status": "ok"}


@app.get("/readiness")
async def readiness():
    return {"status": "ready"}


@app.get("/health", tags=["Health"])
async def health_root():
    return {"status": "ok"}


# Routers (attention aux préfixes attendus par les tests)
# Monte le router health sous /api/v1/health
app.include_router(
    health_routes.router, prefix="/api/v1/health", tags=["Health"]
)
app.include_router(
    main_routes.router, prefix="/api/v1", tags=["Main"]
)
app.include_router(
    estimation_routes.router, prefix="/api/v1", tags=["Estimation"]
)
app.include_router(
    historique_routes.router, prefix="/api/v1/historique", tags=["Historique"]
)

# métriques (optionnel)
app.include_router(metrics_routes.router, tags=["Monitoring"])

# Ajout des gestionnaires d'exceptions avec corrections des types


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        content={"detail": "Validation Error"},
        status_code=422
    )


async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    return JSONResponse(
        content={"detail": "Internal Server Error"},
        status_code=500
    )


# Correction des types passés à add_exception_handler
app.add_exception_handler(
    StarletteHTTPException, http_exception_handler  # type: ignore[arg-type]
)
app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler  # type: ignore[arg-type]
)
app.add_exception_handler(
    Exception, general_exception_handler  # type: ignore[arg-type]
)


