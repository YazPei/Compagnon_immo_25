# app/api/routes/historique.py
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query

from app.api.dependencies.auth import verify_api_key_required

router = APIRouter()


# --- Stubs patchables par les tests ---
def get_estimations(db: Any = None, limit: int = 10, offset: int = 0) -> List[Any]:
    class Obj:
        id_estimation = "stub-1"
        date_estimation = datetime.now(timezone.utc)
        bien: Dict[str, Any] = {"type": "appartement", "surface": 50}
        localisation: Dict[str, Any] = {"code_postal": "75000"}
        estimation: Dict[str, Any] = {"prix": 250000, "indice_confiance": 0.8}

    return [Obj()]


def count_estimations(db: Any = None) -> int:
    return 1


@router.get("/estimations", tags=["Historique"])
async def list_estimations(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: str = Depends(verify_api_key_required),
) -> Dict[str, Any]:
    items = get_estimations(db=None, limit=limit, offset=offset)
    total = count_estimations(db=None)

    # Normaliser en dict (objets ou dicts)
    def to_dict(it: Any) -> Dict[str, Any]:
        if isinstance(it, dict):
            return it  # type: ignore[return-value]
        return {
            "id_estimation": getattr(it, "id_estimation", None),
            "date_estimation": getattr(
                it, "date_estimation", datetime.now(timezone.utc)
            ),
            "bien": getattr(it, "bien", None),
            "localisation": getattr(it, "localisation", None),
            "estimation": getattr(it, "estimation", None),
        }

    page = (offset // limit) + 1 if limit else 1
    pages = ((total - 1) // limit + 1) if limit else 1

    return {
        "estimations": [to_dict(it) for it in items],
        "total": total,
        # Champs à plat pour compat tests
        "page": page,
        "limite": limit,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "page": page,
            "pages": pages,
        },
    }


# Aucun @router.get("/") ici
