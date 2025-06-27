from sqlalchemy.orm import Session
from api_test.app.db.models import EstimationDB
from api_test.app.models.schemas import EstimationRequest, EstimationResponse
from typing import List, Optional
import datetime

# Sauvegarde d'une estimation
def save_estimation(db: Session, id_estimation: str, request: EstimationRequest, response: EstimationResponse):
    db_obj = EstimationDB(
        id_estimation=id_estimation,
        date_estimation=datetime.datetime.fromisoformat(response.metadata.date_estimation),
        bien=request.bien.model_dump(),
        localisation=request.localisation.model_dump(),
        transaction=request.transaction.model_dump(),
        estimation=response.estimation.model_dump(),
        marche=response.marche.model_dump(),
        estimation_metadata=response.metadata.model_dump(),
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

# Récupération d'une estimation par ID
def get_estimation_by_id(db: Session, id_estimation: str) -> Optional[EstimationDB]:
    return db.query(EstimationDB).filter(EstimationDB.id_estimation == id_estimation).first()

# Récupération de l'historique paginé
def get_estimations(db: Session, limit: int = 10, offset: int = 0) -> List[EstimationDB]:
    return db.query(EstimationDB).order_by(EstimationDB.date_estimation.desc()).offset(offset).limit(limit).all()

# Compte total
def count_estimations(db: Session) -> int:
    return db.query(EstimationDB).count() 

def create_estimation(db: Session, request: EstimationRequest, response: EstimationResponse) -> EstimationDB:
    """Crée une nouvelle estimation dans la base de données."""
    db_estimation = EstimationDB(
        bien=request.bien.model_dump(),
        localisation=request.localisation.model_dump(),
        transaction=request.transaction.model_dump(),
        estimation=response.estimation.model_dump(),
        marche=response.marche.model_dump(),
        estimation_metadata=response.metadata.model_dump(),
        id=response.metadata.id,
        date_estimation=response.metadata.date_estimation
    ) 