"""
Configuration de l'application avec support Kubernetes.
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class Settings(BaseSettings):
    """Configuration de l'application avec validation Pydantic."""
    
    # Informations de base
    PROJECT_NAME: str = Field(
        default="Compagnon Immobilier", 
        env="PROJECT_NAME"
    )
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    VERSION: str = Field(default="1.0.0", env="VERSION")  # Alias pour main.py
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Configuration API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=1, env="API_WORKERS")
    API_V1_STR: str = Field(default="/api/v1", env="API_V1_STR")
    
    # Configuration base de données (avec validation)
    DATABASE_URL: str = Field(
        default="sqlite:///./app.db", 
        env="DATABASE_URL",
        description="URL de connexion à la base de données"
    )
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Configuration Redis pour le cache distribué
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0", 
        env="REDIS_URL"
    )
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_SSL: bool = Field(default=False, env="REDIS_SSL")
    
    # Configuration MLflow
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000", 
        env="MLFLOW_TRACKING_URI"
    )
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = Field(
        default=None, 
        env="MLFLOW_S3_ENDPOINT_URL"
    )
    
    # Configuration DVC et DagsHub
    DVC_REMOTE: str = Field(
        default="origin", 
        env="DVC_REMOTE"
    )
    DAGSHUB_URL: str = Field(
        default="https://dagshub.com/YazPei/compagnon_immo.dvc",
        env="DAGSHUB_URL"
    )
    DAGSHUB_USERNAME: Optional[str] = Field(default=None, env="DAGSHUB_USERNAME")
    DAGSHUB_PASSWORD: Optional[str] = Field(default=None, env="DAGSHUB_PASSWORD")
    
    # Configuration sécurité
    API_SECRET_KEY: str = Field(
        default="development-secret-key-change-in-production",
        env="API_SECRET_KEY",
        min_length=32
    )
    
    # Configuration CORS et hosts de confiance
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8501,http://127.0.0.1:3000", 
        env="CORS_ORIGINS"
    )
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8501", 
        env="ALLOWED_ORIGINS"
    )
    TRUSTED_HOSTS: str = Field(default="*", env="TRUSTED_HOSTS")
    
    # Configuration des chemins - Adaptatifs pour Docker/Kubernetes
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # Configuration des logs
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    # Configuration monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")  # Alias pour main.py
    METRICS_PATH: str = Field(default="/metrics", env="METRICS_PATH")
    
    # Configuration Kubernetes
    KUBERNETES_NAMESPACE: Optional[str] = Field(
        default=None, 
        env="KUBERNETES_NAMESPACE"
    )
    KUBERNETES_SERVICE_NAME: Optional[str] = Field(
        default=None, 
        env="KUBERNETES_SERVICE_NAME"
    )
    
    # Configuration de santé
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    READINESS_TIMEOUT: int = Field(default=5, env="READINESS_TIMEOUT")
    LIVENESS_TIMEOUT: int = Field(default=5, env="LIVENESS_TIMEOUT")
    
    # Configuration Airflow
    AIRFLOW_HOME: Optional[str] = Field(default=None, env="AIRFLOW_HOME")
    AIRFLOW_CONN_ID: str = Field(default="compagnon_immo_api", env="AIRFLOW_CONN_ID")

    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Valide que l'environnement est correct."""
        allowed_envs = ["development", "staging", "production", "test"]
        if v not in allowed_envs:
            raise ValueError(f"ENVIRONMENT doit être dans {allowed_envs}")
        return v

    @validator("API_WORKERS")
    def validate_workers(cls, v, values):
        """Valide le nombre de workers selon l'environnement."""
        env = values.get("ENVIRONMENT", "development")
        if env == "development":
            return 1
        elif env == "production" and v < 2:
            return 2  # Minimum 2 workers en production
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configuration des chemins adaptatifs
        self._setup_paths()
        
        # Validation des secrets en production
        if self.ENVIRONMENT == "production":
            self._validate_production_secrets()

    def _setup_paths(self):
        """Configure les chemins selon l'environnement."""
        # Détection de l'environnement d'exécution
        if self.KUBERNETES_NAMESPACE:
            # Kubernetes - Utilise les volumes montés
            self.DATA_DIR = Path("/data")
            self.LOG_DIR = Path("/logs")
            self.MODELS_DIR = Path("/app/api/models")  # Path fixe dans le conteneur
        elif os.path.exists("/.dockerenv"):
            # Docker (non-Kubernetes)
            self.DATA_DIR = Path("/app/data")
            self.LOG_DIR = Path("/app/logs")
            self.MODELS_DIR = Path("/app/api/models")
        elif os.getenv("GITHUB_ACTIONS"):
            # GitHub Actions CI/CD
            self.DATA_DIR = Path("/tmp/data")
            self.LOG_DIR = Path("/tmp/logs")
            self.MODELS_DIR = self.BASE_DIR / "app/api/models"
        else:
            # Développement local
            self.DATA_DIR = self.BASE_DIR / "data"
            self.LOG_DIR = self.BASE_DIR / "logs"
            self.MODELS_DIR = self.BASE_DIR / "app/api/models"
        
        # Créer les répertoires s'ils n'existent pas
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """Crée les répertoires s'ils n'existent pas."""
        directories = [self.DATA_DIR, self.LOG_DIR, self.MODELS_DIR]
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # En mode lecture seule, on ignore l'erreur
                pass

    def _validate_production_secrets(self):
        """Valide que les secrets sont définis en production."""
        if self.API_SECRET_KEY == "development-secret-key-change-in-production":
            raise ValueError(
                "API_SECRET_KEY doit être changée en production"
            )

    @property
    def is_development(self) -> bool:
        """Retourne True si on est en développement."""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Retourne True si on est en production."""
        return self.ENVIRONMENT == "production"

    @property
    def is_kubernetes(self) -> bool:
        """Retourne True si on est dans Kubernetes."""
        return self.KUBERNETES_NAMESPACE is not None

    @property
    def is_docker(self) -> bool:
        """Retourne True si on est dans Docker."""
        return os.path.exists("/.dockerenv")

    @property
    def allowed_origins_list(self) -> List[str]:
        """Retourne la liste des origines autorisées."""
        if not self.ALLOWED_ORIGINS:
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        """Retourne la liste des origines CORS."""
        if not self.CORS_ORIGINS:
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def trusted_hosts_list(self) -> List[str]:
        """Retourne la liste des hosts de confiance."""
        if self.TRUSTED_HOSTS == "*":
            return ["*"]
        return [host.strip() for host in self.TRUSTED_HOSTS.split(",")]

    @property
    def dvc_config(self) -> dict:
        """Configuration DVC pour les conteneurs."""
        return {
            "remote": self.DVC_REMOTE,
            "url": self.DAGSHUB_URL,
            "username": self.DAGSHUB_USERNAME,
            "password": self.DAGSHUB_PASSWORD,
        }

    @property
    def database_config(self) -> dict:
        """Configuration de la base de données."""
        return {
            "url": self.DATABASE_URL,
            "echo": self.DATABASE_ECHO and self.is_development,
            "pool_pre_ping": True if "sqlite" not in self.DATABASE_URL else False,
        }

    def get_model_path(self, model_name: str) -> Path:
        """Retourne le chemin vers un modèle spécifique."""
        return self.MODELS_DIR / f"{model_name}.joblib"

    def get_data_path(self, filename: str) -> Path:
        """Retourne le chemin vers un fichier de données."""
        return self.DATA_DIR / filename

    def get_log_path(self, filename: str = "app.log") -> Path:
        """Retourne le chemin vers un fichier de log."""
        return self.LOG_DIR / filename

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True


# Instance globale des paramètres
settings = Settings()
