"""Routes de health check pour Docker/Kubernetes."""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def health_check() -> Dict[str, str]:
    """Health check basique."""
    return {"status": "healthy", "service": "compagnon-immo"}

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check avec vérification des services."""
    try:
        # Import conditionnel des services
        services_status = {}
        
        try:
            from app.api.services import dvc_service
            services_status["dvc"] = dvc_service is not None
        except:
            services_status["dvc"] = False
            
        try:
            from app.api.services import ml_service
            services_status["ml_models"] = ml_service is not None
        except:
            services_status["ml_models"] = False
            
        return {
            "status": "ready",
            "services": services_status,
            "environment": "docker" if os.path.exists("/.dockerenv") else "local"
        }
    except Exception as e:
        logger.error(f"Erreur readiness check: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Liveness check pour Kubernetes."""
    return {"status": "alive"}