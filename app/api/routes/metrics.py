"""
Endpoint pour exposer les métriques Prometheus.
"""

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.api.monitoring.prometheus_registry import PROMETHEUS_REGISTRY
from app.api.middleware.monitoring import system_metrics_collector

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def get_metrics():
    """
    Endpoint pour exposer les métriques Prometheus.
    
    Returns:
        Response: Métriques au format Prometheus.
    """
    # Collecter les métriques système avant de les exposer
    system_metrics_collector.collect_system_metrics()
    
    # Générer les métriques
    metrics_data = generate_latest(PROMETHEUS_REGISTRY)
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.get("/metrics/health", tags=["Monitoring"])
async def get_metrics_health():
    """
    Health check pour l'endpoint des métriques.
    
    Returns:
        dict: Statut de santé des métriques.
    """
    try:
        # Test de génération des métriques
        generate_latest(PROMETHEUS_REGISTRY)
        
        return {
            "status": "healthy",
            "message": "Métriques Prometheus disponibles",
            "registry_collectors": len(PROMETHEUS_REGISTRY._collector_to_names)
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Erreur lors de la génération des métriques: {str(e)}"
        }