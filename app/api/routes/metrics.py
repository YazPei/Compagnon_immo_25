# app/api/routes/metrics.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/metrics", tags=["Monitoring"])
async def metrics_stub():
    return {"status": "ok"}

