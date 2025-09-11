"""
Application FastAPI principale pour Compagnon Immobilier.
Configuration adaptée pour Kubernetes et environnements cloud.
"""

import os
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

# Imports corrigés avec gestion des erreurs
try:
    from app.api.config.settings import settings
except ImportError as e:
    logging.error(f"Erreur import settings: {e}")
    sys.exit(1)

# Imports conditionnels pour éviter les erreurs
try:
    from app.api.routes import main as main_routes
    from app.api.routes import estimation, health, metrics
except ImportError as e:
    logging.warning(f"Routes manquantes: {e}")
    # Créer des routers vides temporaires
    from fastapi import APIRouter
    main_routes = APIRouter()
    estimation = APIRouter()
    health = APIRouter()
    metrics = APIRouter()

try:
    from app.api.middleware.monitoring import PrometheusMiddleware
    from app.api.middleware.logging import LoggingMiddleware
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    MIDDLEWARE_AVAILABLE = False
    logging.warning("Middleware de monitoring non disponible")

try:
    from app.api.utils.exception_handlers import (
        general_exception_handler,
        http_exception_handler,
        validation_exception_handler
    )
    EXCEPTION_HANDLERS_AVAILABLE = True
except ImportError:
    EXCEPTION_HANDLERS_AVAILABLE = False
    logging.warning("Exception handlers non disponibles")

# Configuration logging robuste
logging.basicConfig(
    level=getattr(logging, getattr(settings, 'LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/logs/app.log') if os.path.exists('/logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application."""
    # Startup
    logger.info(f"🚀 Démarrage de {getattr(settings, 'PROJECT_NAME', 'Compagnon Immobilier')} v{getattr(settings, 'VERSION', '1.0.0')}")
    logger.info(f"🌍 Environnement: {getattr(settings, 'ENVIRONMENT', 'development')}")
    
    # Initialisation des services avec gestion d'erreurs
    try:
        # Vérification des dépendances critiques
        await _initialize_services()
        logger.info("✅ Services initialisés avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        # Ne pas faire sys.exit en production pour éviter les crash loops
        if getattr(settings, 'ENVIRONMENT', 'development') == 'development':
            sys.exit(1)
        else:
            logger.warning("⚠️ Démarrage en mode dégradé")
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt de l'application...")
    # Nettoyage des ressources
    await _cleanup_services()
    logger.info("✅ Nettoyage terminé")


async def _initialize_services():
    """Initialise les services requis."""
    services_status = {}
    
    # Test de connexion base de données
    try:
        from app.api.db.database import check_db_connection
        success, message = check_db_connection()
        services_status['database'] = success
        logger.info(f"Base de données: {message}")
    except Exception as e:
        logger.warning(f"Base de données non disponible: {e}")
        services_status['database'] = False
    
    # Test DVC si disponible
    try:
        from app.api.services.dvc_connector import dvc_connector
        if dvc_connector.is_dvc_available():
            services_status['dvc'] = True
            logger.info("✅ DVC disponible")
        else:
            services_status['dvc'] = False
            logger.warning("⚠️ DVC non disponible")
    except Exception as e:
        logger.warning(f"DVC non disponible: {e}")
        services_status['dvc'] = False
    
    # Chargement des modèles ML
    try:
        from app.api.services import ml_service  # Instance, pas classe
        models_loaded = await ml_service.load_models_from_dvc()
        services_status['ml_models'] = models_loaded.get("models_loaded", 0) > 0
        logger.info(f"Modèles ML chargés: {models_loaded.get('models_loaded', 0)}")
    except Exception as e:
        logger.warning(f"Modèles ML non disponibles: {e}")
        services_status['ml_models'] = False
    
    # Stockage du statut pour les health checks
    app.state.services_status = services_status


async def _cleanup_services():
    """Nettoie les ressources."""
    try:
        # Fermer les connexions DB
        from app.api.db.database import close_db_connections
        close_db_connections()
    except Exception as e:
        logger.error(f"Erreur fermeture DB: {e}")


# Création de l'application FastAPI avec configuration robuste
app = FastAPI(
    title=getattr(settings, 'PROJECT_NAME', 'Compagnon Immobilier'),
    version=getattr(settings, 'VERSION', '1.0.0'),
    description="API d'estimation immobilière avec support Docker/Kubernetes",
    lifespan=lifespan,
    docs_url="/docs" if not getattr(settings, 'is_production', False) else None,
    redoc_url="/redoc" if not getattr(settings, 'is_production', False) else None,
    # Configuration pour les health checks Kubernetes
    openapi_tags=[
        {
            "name": "Health",
            "description": "Endpoints pour les health checks Kubernetes"
        },
        {
            "name": "Estimation",
            "description": "Services d'estimation immobilière"
        },
        {
            "name": "DVC",
            "description": "Gestion des modèles avec DVC"
        }
    ]
)

# Middleware CORS adapté pour tous les environnements
cors_origins = ["*"]  # Default permissif
if hasattr(settings, 'CORS_ORIGINS') and settings.CORS_ORIGINS:
    cors_origins = settings.CORS_ORIGINS.split(",")
elif hasattr(settings, 'allowed_origins_list'):
    cors_origins = settings.allowed_origins_list

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Middleware de monitoring Prometheus (avec gestion d'erreurs)
metrics_enabled = getattr(settings, 'METRICS_ENABLED', True)
if metrics_enabled and MIDDLEWARE_AVAILABLE:
    try:
        app.add_middleware(PrometheusMiddleware)
        
        # Instrumentator Prometheus pour métriques automatiques
        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/metrics", "/readiness", "/liveness"],
            env_var_name="ENABLE_METRICS",
            inprogress_name="fastapi_inprogress",
            inprogress_labels=True
        )
        instrumentator.instrument(app)
        logger.info("✅ Métriques Prometheus activées")
    except Exception as e:
        logger.warning(f"Prometheus non disponible: {e}")

# Middleware de logging (avec fallback)
if MIDDLEWARE_AVAILABLE:
    try:
        app.add_middleware(LoggingMiddleware)
    except Exception as e:
        logger.warning(f"Middleware de logging non disponible: {e}")

# Gestionnaires d'exceptions (avec fallback)
if EXCEPTION_HANDLERS_AVAILABLE:
    try:
        app.add_exception_handler(Exception, general_exception_handler)
        app.add_exception_handler(422, validation_exception_handler)
        app.add_exception_handler(404, http_exception_handler)
    except Exception as e:
        logger.warning(f"Exception handlers non disponibles: {e}")

# Routes avec gestion d'erreurs
try:
    # Health checks (critiques pour Kubernetes)
    app.include_router(
        health.router,
        prefix="/health",
        tags=["Health"]
    )
except Exception as e:
    logger.error(f"Erreur routes health: {e}")
    # Créer un health check minimal
    @app.get("/health")
    async def minimal_health():
        return {"status": "ok", "service": "compagnon-immo"}

try:
    app.include_router(
        estimation.router,
        prefix="/api/v1",
        tags=["Estimation"]
    )
except Exception as e:
    logger.warning(f"Routes estimation non disponibles: {e}")

try:
    app.include_router(
        main_routes.router,
        prefix="/api/v1",
        tags=["Main"]
    )
except Exception as e:
    logger.warning(f"Routes principales non disponibles: {e}")

try:
    # Routes pour les métriques
    app.include_router(
        metrics.router,
        tags=["Monitoring"]
    )
except Exception as e:
    logger.warning(f"Routes métriques non disponibles: {e}")

# Endpoints essentiels pour Kubernetes
@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "service": getattr(settings, 'PROJECT_NAME', 'Compagnon Immobilier'),
        "version": getattr(settings, 'VERSION', '1.0.0'),
        "environment": getattr(settings, 'ENVIRONMENT', 'development'),
        "status": "running",
        "kubernetes_ready": True
    }

@app.get("/readiness")
async def readiness_check():
    """Readiness probe pour Kubernetes."""
    try:
        services_status = getattr(app.state, 'services_status', {})
        # L'API est prête si au moins un service critique fonctionne
        is_ready = services_status.get('database', False) or services_status.get('ml_models', False)
        
        return {
            "status": "ready" if is_ready else "not_ready",
            "services": services_status,
            "timestamp": os.environ.get('HOSTNAME', 'unknown')
        }
    except Exception as e:
        logger.error(f"Erreur readiness check: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/liveness")
async def liveness_check():
    """Liveness probe pour Kubernetes."""
    return {
        "status": "alive",
        "timestamp": os.environ.get('HOSTNAME', 'unknown')
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
    is_development = getattr(settings, 'is_development', True)
    
    if is_development:
        uvicorn.run(
            "app.api.main:app",
            host=getattr(settings, 'API_HOST', '0.0.0.0'),
            port=getattr(settings, 'API_PORT', 8000),
            reload=True,
            log_level=getattr(settings, 'LOG_LEVEL', 'INFO').lower()
        )
    else:
        # En production, utiliser un serveur WSGI externe (gunicorn)
        logger.info(
            "En production, utilisez gunicorn : "
            "gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.api.main:app"
        )