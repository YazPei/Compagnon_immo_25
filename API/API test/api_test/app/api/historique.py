from fastapi import APIRouter, Depends, HTTPException, Query, Header
from api_test.app.models.schemas import HistoriqueResponse, ErrorResponse, HistoriqueItemModel
from api_test.app.security.auth import verify_api_key
from api_test.app.db.database import SessionLocal
from api_test.app.db.crud import get_estimations, count_estimations

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/estimations", response_model=HistoriqueResponse, responses={
    401: {"model": ErrorResponse},
})
def get_estimations_route(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    x_api_key: str = Header(..., alias="X-API-Key"),
    api_key_valid: bool = Depends(verify_api_key),
    db=Depends(get_db)
):
    db_objs = get_estimations(db, limit=limit, offset=offset)
    total = count_estimations(db)
    items = [
        HistoriqueItemModel(
            id_estimation=obj.id_estimation,
            date_estimation=obj.date_estimation.isoformat(),
            bien={
                "type": obj.bien.get("type"),
                "surface": obj.bien.get("surface"),
                "code_postal": obj.localisation.get("code_postal")
            },
            prix_estime=obj.estimation.get("prix"),
            indice_confiance=obj.estimation.get("indice_confiance")
        ) for obj in db_objs
    ]
    return HistoriqueResponse(
        estimations=items,
        estimation_metadata={"total": total, "limit": limit, "offset": offset}
    ) 