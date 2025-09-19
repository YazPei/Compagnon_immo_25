# app/api/routes/main.py
from fastapi import APIRouter, Depends
from app.api.dependencies.auth import verify_api_key

router = APIRouter()

@router.get("/", summary="Ping protégé", tags=["Main"])
async def api_v1_root(_: str = Depends(verify_api_key)):
    return {"status": "ok", "message": "Bienvenue sur l'API v1"}

