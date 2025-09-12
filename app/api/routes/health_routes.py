from fastapi import APIRouter, Request, Response, HTTPException
import logging
from typing import Dict, Any

from app.api.services.health_service import health_service

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
        ml_ready = health_service.health_status != "critical"
        
        if not ml_ready:
            logger.warning("⚠️ Readiness check échoué : ML service non prêt")
            response.status_code = 503
            return {"status": "not_ready", "message": "ML service non prêt"}
        
        logger.info("✅ Readiness check réussi")
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"❌ Erreur lors du readiness check : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne lors du readiness check")


@router.get("/health/complete", status_code=200)
async def complete_health():
    """Health check complet de tous les services."""
    try:
        result = await health_service.comprehensive_check()
        
        if result["status"] == "critical":
            logger.warning("⚠️ Health check complet critique")
            return Response(
                content=result,
                status_code=503,
                media_type="application/json"
            )
        
        logger.info("✅ Health check complet réussi")
        return result
    except Exception as e:
        logger.error(f"❌ Erreur lors du health check complet : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne lors du health check complet")


@router.get("/metrics/system", status_code=200)
async def system_metrics():
    """Métriques système pour monitoring."""
    try:
        # Récupérer les métriques système
        metrics = health_service.check_results.get("system", {})
        logger.info("✅ Récupération des métriques système réussie")
        return metrics
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des métriques système : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la récupération des métriques système")