# app/api/routes/health.py
from typing import Dict

from fastapi import APIRouter

from app.api.services import health_service

router = APIRouter()


@router.get("", summary="Health simple", tags=["Health"])
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/detailed", summary="Health détaillé", tags=["Health"])
async def health_check_detailed() -> Dict[str, object]:
    db = await health_service.check_database()
    ml = await health_service.check_ml_service()
    overall = "healthy"
    if any(c.get("status") != "healthy" for c in (db, ml)):
        overall = "degraded"
    return {
        "status": overall,
        "components": {
            "database": db,
            "ml_service": ml,
        },
    }
