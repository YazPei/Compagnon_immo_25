from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any
import logging

from app.database.session import get_db
from app.database.crud import get_estimations, count_estimations
from ..auth.api_key import verify_api_key
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/historique", tags=["historique"])

@router.get("/estimations")
async def get_estimations_endpoint(
    page: int = Query(1, ge=1, description="Numéro de page"),
    limite: int = Query(10, ge=1, le=100, description="Nombre d'éléments par page"),
    date_debut: Optional[str] = Query(None, description="Date début (YYYY-MM-DD)"),
    date_fin: Optional[str] = Query(None, description="Date fin (YYYY-MM-DD)"),
    type_bien: Optional[str] = Query(None, description="Filtrer par type de bien"),
    ville: Optional[str] = Query(None, description="Filtrer par ville"),
    api_key: str = Depends(verify_api_key),
    db = Depends(get_db)
):
    """
    Récupère l'historique des estimations paginé avec filtres optionnels.
    """
    try:
        offset = (page - 1) * limite
        
        # Construire les filtres
        filters = {}
        if date_debut:
            try:
                filters['date_debut'] = datetime.fromisoformat(date_debut)
            except ValueError:
                raise HTTPException(status_code=400, detail="Format de date_debut invalide (YYYY-MM-DD)")
        
        if date_fin:
            try:
                filters['date_fin'] = datetime.fromisoformat(date_fin)
            except ValueError:
                raise HTTPException(status_code=400, detail="Format de date_fin invalide (YYYY-MM-DD)")
        
        if type_bien:
            filters['type_bien'] = type_bien
        
        if ville:
            filters['ville'] = ville
        
        # Récupérer les estimations avec filtres
        estimations_db = get_estimations(db, limit=limite, offset=offset, filters=filters)
        total = count_estimations(db, filters=filters)
        
        # Formatter les résultats
        estimations = []
        for est in estimations_db:
            estimation_data = {
                "id_estimation": est.id_estimation,
                "date_estimation": est.date_estimation.isoformat() if hasattr(est.date_estimation, 'isoformat') else str(est.date_estimation),
                "bien": est.bien if hasattr(est, 'bien') else {},
                "localisation": est.localisation if hasattr(est, 'localisation') else {},
                "estimation": est.estimation if hasattr(est, 'estimation') else {}
            }
            
            # Ajouter des champs supplémentaires si disponibles
            if hasattr(est, 'prix_estime'):
                estimation_data["prix_estime"] = est.prix_estime
            if hasattr(est, 'confiance'):
                estimation_data["confiance"] = est.confiance
            if hasattr(est, 'modele_utilise'):
                estimation_data["modele_utilise"] = est.modele_utilise
                
            estimations.append(estimation_data)
        
        return {
            "estimations": estimations,
            "total": total,
            "page": page,
            "limite": limite,
            "total_pages": (total + limite - 1) // limite,
            "filters_applied": filters
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@router.get("/estimations/{estimation_id}")
async def get_estimation_detail(
    estimation_id: str,
    api_key: str = Depends(verify_api_key),
    db = Depends(get_db)
):
    """
    Récupère le détail d'une estimation spécifique.
    """
    try:
        estimation = get_estimation_by_id(db, estimation_id)
        
        if not estimation:
            raise HTTPException(status_code=404, detail="Estimation non trouvée")
        
        return {
            "id_estimation": estimation.id_estimation,
            "date_estimation": estimation.date_estimation.isoformat() if hasattr(estimation.date_estimation, 'isoformat') else str(estimation.date_estimation),
            "bien": estimation.bien,
            "localisation": estimation.localisation,
            "estimation": estimation.estimation,
            "details": {
                "prix_estime": getattr(estimation, 'prix_estime', None),
                "confiance": getattr(estimation, 'confiance', None),
                "modele_utilise": getattr(estimation, 'modele_utilise', None),
                "features_utilisees": getattr(estimation, 'features_utilisees', {}),
                "comparables": getattr(estimation, 'comparables', [])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'estimation {estimation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@router.get("/stats")
async def get_historique_stats(
    periode: str = Query("30d", description="Période (7d, 30d, 90d, 1y)"),
    api_key: str = Depends(verify_api_key),
    db = Depends(get_db)
):
    """
    Récupère les statistiques de l'historique des estimations.
    """
    try:
        # Définir la période
        period_mapping = {
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "1y": 365
        }
        
        if periode not in period_mapping:
            raise HTTPException(status_code=400, detail="Période invalide (7d, 30d, 90d, 1y)")
        
        days = period_mapping[periode]
        date_limite = datetime.now() - timedelta(days=days)
        
        # Récupérer les stats
        stats = get_estimation_stats(db, date_limite)
        
        return {
            "periode": periode,
            "date_debut": date_limite.isoformat(),
            "total_estimations": stats.get("total", 0),
            "prix_moyen": stats.get("prix_moyen", 0),
            "prix_median": stats.get("prix_median", 0),
            "repartition_types": stats.get("repartition_types", {}),
            "repartition_villes": stats.get("repartition_villes", {}),
            "evolution_mensuelle": stats.get("evolution_mensuelle", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du calcul des stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")
