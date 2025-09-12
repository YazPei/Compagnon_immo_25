from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class PropertyBase(BaseModel):
    property_type: str
    surface: float
    postal_code: str
    

class PropertyCreate(PropertyBase):
    price: Optional[float] = None


class Property(PropertyBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


class EstimationBase(BaseModel):
    property_data: Dict[str, Any]
    estimated_price: float
    confidence_score: float


class EstimationCreate(EstimationBase):
    model_name: Optional[str] = None


class Estimation(EstimationBase):
    id: int
    model_name: str
    mlflow_run_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        orm_mode = True
        