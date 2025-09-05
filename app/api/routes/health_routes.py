from fastapi import APIRouter, Request, Response
import time
import logging
import psutil
import os
from typing import Dict, Any

from app.services.health_service import health_service
from app.services.ml_service import ml_service
from app.services.dvc_connector import dvc_connector

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", status_code=200)
async def health():
    """Health check basique."""
    return {"status": "healthy", "timestamp": time.time()}


@router.get("/health/live", status_code=200)
async def liveness():
    """Liveness probe pour Kubernetes."""
    return {"status": "alive", "timestamp": time.time()}


@router.get("/health/ready", status_code=200)
async def readiness(response: Response):
    """Readiness probe pour Kubernetes."""
    try:
        # Vérification des services critiques
        ml_status = ml_service.get_status()
        ml_ready = ml_status.get("status") != "error"
        
        # Vérification DVC si disponible
        dvc_ready = True
        try:
            dvc_status = dvc_connector.check_models_status()
            dvc_ready = dvc_status.get("status") in ["success", "warning"]
        except Exception:
            dvc_ready = False
        
        if not ml_ready or not dvc_ready:
            response.status_code = 503
            return {
                "status": "not_ready", 
                "ml_ready": ml_ready,
                "dvc_ready": dvc_ready,
                "timestamp": time.time()
            }
        
        return {
            "status": "ready",
            "ml_ready": ml_ready,
            "dvc_ready": dvc_ready,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Erreur readiness check: {str(e)}")
        response.status_code = 503
        return {"status": "not_ready", "error": str(e), "timestamp": time.time()}


@router.get("/health/complete")
async def complete_health(response: Response):
    """Health check complet de tous les services."""
    try:
        result = {
            "status": "healthy",
            "timestamp": time.time(),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "services": {}
        }
        
        # Check ML Service
        try:
            ml_status = ml_service.get_status()
            result["services"]["ml"] = ml_status
        except Exception as e:
            result["services"]["ml"] = {"status": "error", "message": str(e)}
        
        # Check DVC
        try:
            dvc_status = dvc_connector.check_models_status()
            result["services"]["dvc"] = dvc_status
        except Exception as e:
            result["services"]["dvc"] = {"status": "error", "message": str(e)}
        
        # Check système
        result["services"]["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "status": "healthy"
        }
        
        # Check configuration DagsHub
        result["services"]["dagshub"] = {
            "configured": bool(os.getenv('DAGSHUB_USERNAME') and os.getenv('DAGSHUB_TOKEN')),
            "username": os.getenv('DAGSHUB_USERNAME', 'not_set'),
            "status": "configured" if os.getenv('DAGSHUB_USERNAME') else "not_configured"
        }
        
        # Déterminer le statut global
        service_statuses = [s.get("status", "error") for s in result["services"].values()]
        if any(status == "error" for status in service_statuses):
            result["status"] = "degraded"
            response.status_code = 503
        elif any(status == "warning" for status in service_statuses):
            result["status"] = "warning"
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur health check complet: {str(e)}")
        response.status_code = 503
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get("/metrics/system")
async def system_metrics():
    """Métriques système pour monitoring."""
    try:
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Erreur métriques système: {str(e)}")
        return {"error": str(e), "timestamp": time.time()}


@router.get("/metrics/models")
async def models_metrics():
    """Métriques des modèles ML."""
    try:
        ml_status = ml_service.get_status()
        dvc_status = dvc_connector.check_models_status()
        
        return {
            "ml_service": ml_status,
            "dvc_status": dvc_status,
            "models_loaded": ml_status.get("models_loaded", 0),
            "last_reload": ml_status.get("last_reload"),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Erreur métriques modèles: {str(e)}")
        return {"error": str(e), "timestamp": time.time()}