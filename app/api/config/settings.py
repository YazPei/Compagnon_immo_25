from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Configuration de l'application."""
    
    # Application
    APP_NAME: str = "Compagnon Immo API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API
    API_PREFIX: str = "/api/v1"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Compagnon Immobilier API"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Sécurité
    SECRET_KEY: str = os.getenv("SECRET_KEY", "compagnon-immo-secret-2024")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "compagnon-immo-jwt-secret-2024")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_KEYS: List[str] = ["dev-api-key", "test-api-key"]
    
    # Base de données
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/compagnon_immo.db")
    DATABASE_ECHO: bool = False
    
    # MLflow et DVC - CORRIGÉ
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", 
        "https://dagshub.com/YazPei/compagnon_immo.mlflow"
    )
    MLFLOW_EXPERIMENT_NAME: str = "compagnon-immo-production"
    DVC_REMOTE_URL: str = os.getenv("DVC_REMOTE_URL", "")
    DAGSHUB_TOKEN: str = os.getenv("DAGSHUB_TOKEN", "")
    DAGSHUB_USERNAME: str = os.getenv("DAGSHUB_USERNAME", "YazPei")
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000"
    ]
    CORS_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/api.log"
    
    # Modèles et données
    MODELS_PATH: str = "app/api/models"
    DATA_PATH: str = "data"
    CSV_SALES_PATH: str = "df_sales_clean_with_cluster.csv"
    MODEL_CACHE_SIZE: int = 10
    
    # Limites
    REQUEST_TIMEOUT: int = 30
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Instance globale
settings = Settings()


# Validation des chemins
def validate_paths():
    """Valide et crée les chemins nécessaires."""
    paths_to_create = [
        Path(settings.DATA_PATH),
        Path(settings.MODELS_PATH),
        Path("logs"),
        Path("cache")
    ]
    
    for path in paths_to_create:
        path.mkdir(parents=True, exist_ok=True)


# Auto-validation au import
validate_paths()