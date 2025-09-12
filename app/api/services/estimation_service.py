"""Service d'estimation immobilière."""

from app.api.models.schemas import EstimationRequest, EstimationResponse
from app.api.services.estimation_logic import compute_estimation
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_property(
    estimation_request: EstimationRequest, 
    enriched_features: Optional[Dict[str, Any]] = None,
    debug_info: Optional[Dict[str, Any]] = None
) -> EstimationResponse:
    """
    Service principal pour estimer un bien immobilier.
    
    Args:
        estimation_request: Requête d'estimation
        enriched_features: Features enrichies (optionnel)
        debug_info: Informations de debug (optionnel)
    
    Returns:
        EstimationResponse: Réponse contenant les résultats de l'estimation.
    """
    try:
        logger.info("🔄 Début de l'estimation d'un bien immobilier...")
        
        # Log des informations de debug si disponibles
        if debug_info:
            logger.debug(f"Debug info: {debug_info}")
        
        # Appel à la logique principale d'estimation
        response = compute_estimation(estimation_request)
        
        logger.info("✅ Estimation réalisée avec succès.")
        return response
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'estimation : {str(e)}")
        raise Exception(f"Erreur lors de l'estimation : {str(e)}")