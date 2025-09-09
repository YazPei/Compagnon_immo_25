"""
Configuration de l'application adaptée pour Kubernetes.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuration de l'application avec support Kubernetes."""
    
    # Configuration de base
    PROJECT_NAME: str = Field(default="Compagnon Immobilier API", env="PROJECT_NAME")
    VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Configuration serveur
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")  # Bind sur toutes interfaces pour K8s
    API_PORT: int = Field(default=8000, env="API_PORT")
    
    # Mode debug basé sur l'environnement
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    @property
    def is_development(self) -> bool:
        """Vérifie si on est en mode développement."""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Vérifie si on est en mode production."""
        return self.ENVIRONMENT.lower() == "production"
    
    # Configuration base de données
    DATABASE_URL: str = Field(
        default="sqlite:///./app.db",
        env="DATABASE_URL",
        description="URL de connexion à la base de données"
    )
    
    # Configuration Redis pour cache distribué
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="URL Redis pour le cache distribué"
    )
    
    # Configuration MLflow
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI",
        description="URI du serveur MLflow"
    )
    
    # Configuration des modèles pour stockage distribué
    MODEL_REGISTRY_URL: Optional[str] = Field(
        default=None,
        env="MODEL_REGISTRY_URL",
        description="URL du registre de modèles (S3, GCS, etc.)"
    )
    
    # Configuration logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json pour K8s
    
    # Configuration workers (pour production)
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Configuration timeouts
    REQUEST_TIMEOUT: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Configuration health checks
    HEALTH_CHECK_TIMEOUT: int = Field(default=5, env="HEALTH_CHECK_TIMEOUT")
    
    # Configuration métriques
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # Configuration sécurité
    API_KEY: Optional[str] = Field(default=None, env="API_KEY")
    CORS_ORIGINS: str = Field(default="*", env="CORS_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
