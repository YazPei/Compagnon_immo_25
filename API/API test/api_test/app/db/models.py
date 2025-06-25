from sqlalchemy import Column, String, Float, Integer, DateTime, JSON
from api_test.app.db.database import Base
import datetime

class EstimationDB(Base):
    __tablename__ = "estimations"

    id_estimation = Column(String, primary_key=True, index=True)
    date_estimation = Column(DateTime, default=datetime.datetime.utcnow)
    bien = Column(JSON, nullable=False)
    localisation = Column(JSON, nullable=False)
    transaction = Column(JSON, nullable=False)
    estimation = Column(JSON, nullable=False)
    marche = Column(JSON, nullable=False)
    estimation_metadata = Column(JSON, nullable=False) 