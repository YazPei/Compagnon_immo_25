from fastapi import APIRouter, Depends, Query, Header, HTTPException
from typing import Optional, List, Dict, Any
from app.api.db.database import SessionLocal
from app.api.db.crud import get_estimations, count_estimations
from app.api.security.auth import verify_api_key
from datetime import datetime
import logging

router = APIRouter(prefix="/api/v1", tags=["historique"])
logger = logging.getLogger(__name__)


def get_db():
    """Dependency pour obtenir une session de base de données."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/historique/estimations", response_model=Dict[str, Any])
def get_estimations_endpoint(
    page: int = Query(1, ge=1, description="Numéro de page"),
    limite: int = Query(10, ge=1, le=100, description="Nombre d'éléments par page"),
    x_api_key: str = Header(..., alias="X-API-Key"),
    api_key_valid: bool = Depends(verify_api_key),
    db=Depends(get_db)
):
    """
    Récupère l'historique des estimations paginé.
    """
    try:
        logger.info(f"🔄 Récupération des estimations (page={page}, limite={limite})")

        offset = (page - 1) * limite
        
        # Récupérer les estimations
        estimations_db = get_estimations(db, limit=limite, offset=offset)
        total = count_estimations(db)
        
        # Formatter les résultats
        estimations = []
        for est in estimations_db:
            estimations.append({
                "id_estimation": est.id_estimation,
                "date_estimation": est.date_estimation.isoformat() if hasattr(est.date_estimation, 'isoformat') else str(est.date_estimation),
                "bien": est.bien,
                "localisation": est.localisation,
                "estimation": est.estimation
            })
        
        logger.info(f"✅ {len(estimations)} estimations récupérées sur un total de {total}")
        
        return {
            "estimations": estimations,
            "total": total,
            "page": page,
            "limite": limite,
            "total_pages": (total + limite - 1) // limite
        }
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des estimations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")