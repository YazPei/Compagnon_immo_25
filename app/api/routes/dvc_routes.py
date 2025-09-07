from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from app.services.dvc_connector import dvc_connector
from app.services.ml_service import ml_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/dvc", tags=["dvc"])

@router.get("/status")
async def get_dvc_status():
    """Voir l'état de DVC et des modèles."""
    try:
        status = dvc_connector.check_models_status()
        logger.info("✅ État de DVC récupéré avec succès")
        return status
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération de l'état de DVC: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de l'état de DVC")

@router.post("/pull")
async def pull_dvc_models():
    """Récupérer les derniers modèles depuis DVC."""
    try:
        logger.info("🔄 Début de la synchronisation des modèles avec DVC...")
        result = dvc_connector.pull_latest_models()

        if result["status"] in ["success", "warning"]:
            logger.info("🔄 Rechargement des modèles ML après synchronisation DVC...")
            ml_result = ml_service.refresh_models()
            result["ml_reload"] = ml_result
            logger.info("✅ Modèles ML rechargés avec succès")
        else:
            logger.warning("⚠️ Synchronisation DVC terminée avec des avertissements")

        return result
    except Exception as e:
        logger.error(f"❌ Erreur lors de la synchronisation des modèles avec DVC: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la synchronisation des modèles avec DVC")

@router.get("/models/status")
async def get_models_status():
    """État des modèles ML chargés."""
    try:
        status = ml_service.get_status()
        logger.info("✅ État des modèles ML récupéré avec succès")
        return status
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération de l'état des modèles ML: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de l'état des modèles ML")

@router.post("/models/reload")
async def reload_models():
    """Recharger les modèles ML."""
    try:
        logger.info("🔄 Rechargement des modèles ML...")
        result = ml_service.refresh_models()
        logger.info("✅ Modèles ML rechargés avec succès")
        return result
    except Exception as e:
        logger.error(f"❌ Erreur lors du rechargement des modèles ML: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors du rechargement des modèles ML")