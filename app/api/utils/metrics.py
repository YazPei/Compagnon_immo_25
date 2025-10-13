from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, generate_latest

router = APIRouter()

# Définir des métriques Prometheus
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Nombre total de requêtes HTTP",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP en secondes",
    ["method", "endpoint"],
)


@router.get("/metrics")
async def prometheus_metrics():
    """Expose les métriques Prometheus."""
    return Response(generate_latest(), media_type="text/plain")
