"""
Application FastAPI principale avec métriques centralisées.
"""

import logging
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.api.config.settings import settings
from app.api.routes import main as main_routes
from app.api.routes import estimation, health, metrics
from app.api.middleware.monitoring import PrometheusMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.utils.exception_handlers import (
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler
)

# Configuration logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    # Startup
    logger.info(f"🚀 Démarrage de {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"🌍 Environnement: {settings.ENVIRONMENT}")
    
    # Initialisation des services
    try:
        # Ici vous pouvez initialiser vos services (DB, ML models, etc.)
        logger.info("✅ Services initialisés avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt de l'application...")
    # Nettoyage des ressources
    logger.info("✅ Nettoyage terminé")


# Création de l'application FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API d'estimation immobilière",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de monitoring Prometheus (utilisant le registry centralisé)
if settings.METRICS_ENABLED:
    app.add_middleware(PrometheusMiddleware)

# Middleware de logging
app.add_middleware(LoggingMiddleware)

# Gestionnaires d'exceptions
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(422, validation_exception_handler)
app.add_exception_handler(404, http_exception_handler)

# Routes
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    estimation.router,
    prefix="/api/v1",
    tags=["Estimation"]
)

app.include_router(
    main_routes.router,
    prefix="/api/v1",
    tags=["Main"]
)

# Routes pour les métriques
app.include_router(
    metrics.router,
    tags=["Monitoring"]
)


@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "running"
    }


# Gestionnaire de signaux pour arrêt gracieux
def signal_handler(signum, frame):
    """Gestionnaire pour arrêt gracieux."""
    logger.info(f"Signal {signum} reçu, arrêt en cours...")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    # Configuration pour développement local uniquement
    if settings.is_development:
        uvicorn.run(
            "app.api.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=True,
            log_level=settings.LOG_LEVEL.lower()
        )
    else:
        # En production, utiliser un serveur WSGI externe (gunicorn)
        logger.warning(
            "En production, utilisez un serveur WSGI comme gunicorn"
        )