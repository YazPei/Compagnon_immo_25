# app/api/routes/historique.py
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, Query

from app.api.dependencies.auth import verify_api_key_required

router = APIRouter()

# --- Stubs patchables par les tests ---
def get_estimations(db=None, limit: int = 10, offset: int = 0):
    class Obj:
        id_estimation = "stub-1"
        date_estimation = datetime.utcnow()
        bien = {"type": "appartement", "surface": 50}
        localisation = {"code_postal": "75000"}
        estimation = {"prix": 250000, "indice_confiance": 0.8}
    return [Obj()]

def count_estimations(db=None) -> int:
    return 1

@router.get("/estimations", tags=["Historique"])
async def list_estimations(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: str = Depends(verify_api_key_required),
):
    items = get_estimations(db=None, limit=limit, offset=offset)
    total = count_estimations(db=None)

    # Normaliser en dict (objets ou dicts)
    def to_dict(it):
        if isinstance(it, dict):
            return it
        return {
            "id_estimation": getattr(it, "id_estimation", None),
            "date_estimation": getattr(it, "date_estimation", datetime.utcnow()),
            "bien": getattr(it, "bien", None),
            "localisation": getattr(it, "localisation", None),
            "estimation": getattr(it, "estimation", None),
        }

    page = (offset // limit) + 1 if limit else 1
    pages = ((total - 1) // limit + 1) if limit else 1

    return {
        "estimations": [to_dict(it) for it in items],
        "total": total,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "page": page,
            "pages": pages,
        },
    }

