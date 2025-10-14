# app/api/db/models.py
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

# ⚠️ On utilise le SEUL Base du projet (défini dans database.py)
from app.api.db.database import Base


class Property(Base):
    """Modèle pour les propriétés immobilières."""

    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(255), nullable=False)
    postal_code = Column(String(10), nullable=False, index=True)
    city = Column(String(100), nullable=False)
    property_type = Column(String(50), nullable=False)
    surface = Column(Float, nullable=False)
    rooms = Column(Integer)
    price = Column(Float, nullable=False)
    price_per_m2 = Column(Float)
    longitude = Column(Float)
    latitude = Column(Float)
    cluster_id = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Estimation(Base):
    """Modèle pour les estimations utilisateur."""

    __tablename__ = "estimations"

    id = Column(Integer, primary_key=True, index=True)
    # JSON sérialisé (string) des caractéristiques
    property_data = Column(Text, nullable=False)
    estimated_price = Column(Float, nullable=False)
    confidence_score = Column(Float)

    # Référence au run MLflow (au lieu de dupliquer des infos de modèle)
    mlflow_run_id = Column(String(255), index=True)
    model_name = Column(String(100))

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class APIKey(Base):
    """Modèle pour les clés API."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String(255), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True))


# --- Modèle minimal User pour satisfaire les imports/tests ---
class User(Base):
    """Utilisateur minimal (pour tests et dépendances)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
