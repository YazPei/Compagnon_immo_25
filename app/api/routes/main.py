# app/api/routes/main.py
from fastapi import APIRouter, Depends

from app.api.utils.auth import get_api_key  # adapte ce chemin si besoin

router = APIRouter()


@router.get("/")
async def api_v1_root(api_key: str = Depends(get_api_key)):
    return {"message": "API v1 root OK"}
