from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from app.services.dvc_connector import dvc_connector
from app.services.ml_service import ml_service
from ..auth.api_key import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/dvc", tags=["dvc"])

@router.get("/status")
async def get_dvc_status(api_key: str = Depends(verify_api_key)):
    """Voir l'état de DVC et des modèles."""
    try:
        return dvc_connector.check_models_status()
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du statut DVC: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pull")
async def pull_dvc_models(api_key: str = Depends(verify_api_key)):
    """Récupérer les derniers modèles depuis DVC."""
    try:
        result = dvc_connector.pull_latest_models()
        
        # Recharger les modèles ML si le pull s'est bien passé
        if result.get("status") in ["success", "warning"]:
            try:
                ml_result = ml_service.refresh_models()
                result["ml_reload"] = ml_result
                logger.info("Modèles ML rechargés après pull DVC")
            except Exception as ml_error:
                logger.warning(f"Erreur lors du rechargement des modèles ML: {str(ml_error)}")
                result["ml_reload"] = {"status": "error", "message": str(ml_error)}
        
        return result
    except Exception as e:
        logger.error(f"Erreur lors du pull DVC: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/push")
async def push_dvc_models(api_key: str = Depends(verify_api_key)):
    """Pousser les modèles locaux vers DVC."""
    try:
        result = dvc_connector.push_models()
        return result
    except Exception as e:
        logger.error(f"Erreur lors du push DVC: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status")
async def get_models_status():
    """État des modèles ML chargés (endpoint public pour monitoring)."""
    try:
        return ml_service.get_status()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut des modèles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/reload")
async def reload_models(api_key: str = Depends(verify_api_key)):
    """Recharger les modèles ML."""
    try:
        result = ml_service.refresh_models()
        logger.info("Modèles ML rechargés manuellement")
        return result
    except Exception as e:
        logger.error(f"Erreur lors du rechargement des modèles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))