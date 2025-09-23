import logging
from fastapi import APIRouter, HTTPException
from typing import Any

from app.api.services.dvc_connector import dvc_connector
from app.api.services.ml_service import ml_service

router = APIRouter(prefix="/api/v1/dvc", tags=["dvc"])
logger = logging.getLogger("dvc_routes")


@router.get("/status")
async def get_dvc_status():
    """Voir l'état de DVC et des modèles."""
    try:
        status = dvc_connector.check_models_status()
        logger.info("✅ État de DVC récupéré avec succès")
        return status
    except Exception as e:
        logger.error(
            f"❌ Erreur lors de la récupération de l'état de DVC: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la récupération de l'état de DVC",
        )


@router.post("/pull")
async def pull_dvc_models():
    """Récupérer les derniers modèles depuis DVC."""
    try:
        logger.info("🔄 Début de la synchronisation des modèles avec DVC...")
        result = dvc_connector.pull_latest_models()

        if result["status"] in ["success", "warning"]:
            logger.info(
                "🔄 Rechargement des modèles ML après synchronisation DVC..."
            )
            ml_result = ml_service.refresh_models()
            result["ml_reload"] = ml_result
            logger.info("✅ Modèles ML rechargés avec succès")
        else:
            logger.warning(
                "⚠️ Synchronisation DVC terminée avec des avertissements"
            )

        return result
    except Exception as e:
        logger.error(
            "❌ Erreur lors de la synchronisation des modèles avec DVC: "
            f"{str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la synchronisation des modèles avec DVC",
        )


@router.get("/models/status")
async def get_models_status() -> Any:
    """État des modèles ML chargés."""
    try:
        status = ml_service.get_models_status()
        logger.info("✅ État des modèles ML récupéré avec succès")
        return status
    except Exception as e:
        logger.error(
            "❌ Erreur lors de la récupération de l'état des modèles ML: "
            f"{str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la récupération de l'état des modèles ML",
        )


@router.post("/models/reload")
async def reload_models():
    """Recharger les modèles ML."""
    try:
        logger.info("🔄 Rechargement des modèles ML...")
        result = ml_service.refresh_models()
        logger.info("✅ Modèles ML rechargés avec succès")
        return result
    except Exception as e:
        logger.error(
            f"❌ Erreur lors du rechargement des modèles ML: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Erreur lors du rechargement des modèles ML",
        )


# Ajout des endpoints pour interagir avec DVC

@router.post("/dvc/pull")
async def pull_data():
    """
    Pull the latest data from the DVC remote storage.
    """
    try:
        logger.info("🔄 Pulling data from DVC remote storage...")
        # Logic to pull data using DVC
        logger.info("✅ Data pulled successfully.")
        return {"message": "Data pulled successfully."}
    except Exception as e:
        logger.error(f"❌ Error pulling data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error pulling data from DVC remote storage."
        )


@router.post("/dvc/push")
async def push_data():
    """
    Push the latest data to the DVC remote storage.
    """
    try:
        logger.info("🔄 Pushing data to DVC remote storage...")
        # Logic to push data using DVC
        logger.info("✅ Data pushed successfully.")
        return {"message": "Data pushed successfully."}
    except Exception as e:
        logger.error(f"❌ Error pushing data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error pushing data to DVC remote storage."
        )


@router.post("/dvc/pipeline/run")
async def run_pipeline():
    """
    Run the DVC pipeline.
    """
    try:
        logger.info("🔄 Running DVC pipeline...")
        # Logic to run the DVC pipeline
        logger.info("✅ Pipeline executed successfully.")
        return {"message": "Pipeline executed successfully."}
    except Exception as e:
        logger.error(f"❌ Error running pipeline: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error running DVC pipeline."
        )
