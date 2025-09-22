from fastapi import APIRouter, Response, HTTPException
import logging
from typing import Dict, Any

from app.api.services.health_service import check_database, check_ml_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", status_code=200)
async def health():
    """Health check basique."""
    logger.info("✅ Health check basique effectué")
    return {"status": "healthy"}


@router.get("/health/live", status_code=200)
async def liveness():
    """Liveness probe pour Kubernetes."""
    logger.info("✅ Liveness check effectué")
    return {"status": "alive"}


@router.get("/health/ready", status_code=200)
async def readiness(response: Response):
    """Readiness probe pour Kubernetes."""
    try:
        # Vérification de base des services critiques
        db_ready = await check_database()
        ml_ready = await check_ml_service()

        if db_ready["status"] != "healthy" or ml_ready["status"] != "healthy":
            logger.warning("⚠️ Readiness check échoué")
            response.status_code = 503
            return {"status": "not_ready", "message": "Services non prêts"}

        logger.info("✅ Readiness check réussi")
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"❌ Erreur lors du readiness check : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors du readiness check",
        )


@router.get("/health/complete", status_code=200)
async def complete_health() -> Dict[str, Any]:
    """Health check complet de tous les services."""
    try:
        db = await check_database()
        ml = await check_ml_service()

        overall = "healthy"
        if db["status"] != "healthy" or ml["status"] != "healthy":
            overall = "degraded"

        logger.info("✅ Health check complet réussi")
        return {
            "status": overall,
            "components": {
                "database": db,
                "ml_service": ml,
            },
        }
    except Exception as e:
        logger.error(f"❌ Erreur lors du health check complet : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors du health check complet",
        )
