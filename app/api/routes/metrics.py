# app/api/routes/metrics.py
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest

from app.api.monitoring.prometheus_registry import PROMETHEUS_REGISTRY

router = APIRouter()


@router.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(PROMETHEUS_REGISTRY), media_type="text/plain")
