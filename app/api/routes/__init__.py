"""Routes de l'application avec imports conditionnels."""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Routes par défaut si les modules spécialisés ne sont pas disponibles
def create_empty_router():
    """Crée un router vide pour les fallbacks."""
    return APIRouter()

# Import conditionnel des routes
try:
    from . import estimation
    logger.info("✅ Routes estimation importées")
except ImportError as e:
    logger.warning(f"Routes estimation non disponibles: {e}")
    estimation = create_empty_router()

try:
    from . import health
    logger.info("✅ Routes health importées")
except ImportError as e:
    logger.warning(f"Routes health non disponibles: {e}")
    health = create_empty_router()
    
    # Health check minimal
    @health.get("/")
    async def basic_health():
        return {"status": "ok", "service": "compagnon-immo"}

try:
    from . import metrics
    logger.info("✅ Routes metrics importées")
except ImportError as e:
    logger.warning(f"Routes metrics non disponibles: {e}")
    metrics = create_empty_router()

try:
    from . import main
    logger.info("✅ Routes main importées")
except ImportError as e:
    logger.warning(f"Routes main non disponibles: {e}")
    main = create_empty_router()

try:
    from . import historique
    logger.info("✅ Routes historique importées")
except ImportError as e:
    logger.warning(f"Routes historique non disponibles: {e}")
    historique = create_empty_router()

__all__ = ['estimation', 'health', 'metrics', 'main', 'historique']