# app/api/config/settings.py
"""
Configuration centralisée (Pydantic v2 + pydantic-settings).
Inclut alias JWT, rate limiting et chemins adaptatifs (Docker/K8S/CI/local).
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Pydantic v2 settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
        validate_assignment=True,
        extra="ignore",
    )

    # --- Base ---
    PROJECT_NAME: str = "Compagnon Immobilier"
    APP_VERSION: str = "1.0.0"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_V1_STR: str = "/api/v1"

    # --- Sécurité ---
    # IMPORTANT: Les tests attendent X-API-Key == "test_api_key"
    API_KEY: str = "test_api_key"

    # --- DB ---
    DATABASE_URL: str = "sqlite:///./app.db"
    DATABASE_ECHO: bool = False

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SSL: bool = False

    # --- MLflow ---
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = None

    # --- DVC / DagsHub ---
    DVC_REMOTE: str = "origin"
    DAGSHUB_URL: str = "https://dagshub.com/YazPei/compagnon_immo.dvc"
    DAGSHUB_USERNAME: Optional[str] = None
    DAGSHUB_PASSWORD: Optional[str] = None

    # --- JWT ---
    API_SECRET_KEY: str = "development-secret-key-change-in-production"
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- Rate limiting ---
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # en secondes
    RATE_LIMIT_WINDOW_SECONDS: int = 60  # alias

    # --- CORS / hosts ---
    CORS_ORIGINS: str = (
        "http://localhost:3000,http://localhost:8501,http://127.0.0.1:3000"
    )
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8501"
    TRUSTED_HOSTS: str = "*"

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = Path("./models")
    DATA_DIR: Path = Path("./data")
    LOG_DIR: Path = Path("./logs")

    # --- Logs ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # --- Monitoring ---
    ENABLE_METRICS: bool = True
    METRICS_ENABLED: bool = True
    METRICS_PATH: str = "/metrics"

    # --- Kubernetes ---
    KUBERNETES_NAMESPACE: Optional[str] = None
    KUBERNETES_SERVICE_NAME: Optional[str] = None

    # --- Santé ---
    HEALTH_CHECK_INTERVAL: int = 30
    READINESS_TIMEOUT: int = 5
    LIVENESS_TIMEOUT: int = 5

    # --- Airflow ---
    AIRFLOW_HOME: Optional[str] = None
    AIRFLOW_CONN_ID: str = "compagnon_immo_api"

    # Ajout d'un modèle pour valider les champs imbriqués si nécessaire
    class Config:
        validate_assignment = True
        extra = "ignore"

    # -------- Validators / Hooks --------
    @field_validator("ENVIRONMENT")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT doit être dans {sorted(allowed)}")
        return v

    @field_validator("REDIS_URL", mode="before")
    def validate_redis_url(cls, value: str) -> str:
        if not value.startswith("redis://"):
            raise ValueError("REDIS_URL doit commencer par 'redis://'.")
        return value

    @field_validator("MLFLOW_TRACKING_URI", mode="before")
    def validate_mlflow_tracking_uri(cls, value: str) -> str:
        if not value.startswith("http"):
            raise ValueError("MLFLOW_TRACKING_URI doit commencer par 'http'.")
        return value

    @model_validator(mode="after")
    def _after(self) -> "Settings":
        # alias JWT: si non fourni, reprendre API_SECRET_KEY
        if not self.JWT_SECRET_KEY:
            object.__setattr__(self, "JWT_SECRET_KEY", self.API_SECRET_KEY)

        # assurer cohérence des deux noms de fenêtre RL
        if self.RATE_LIMIT_WINDOW <= 0 and self.RATE_LIMIT_WINDOW_SECONDS > 0:
            object.__setattr__(
                self, "RATE_LIMIT_WINDOW", self.RATE_LIMIT_WINDOW_SECONDS
            )
        elif (
            self.RATE_LIMIT_WINDOW > 0
            and self.RATE_LIMIT_WINDOW_SECONDS <= 0
        ):
            object.__setattr__(
                self, "RATE_LIMIT_WINDOW_SECONDS", self.RATE_LIMIT_WINDOW
            )

        # Ajustement workers
        if self.ENVIRONMENT == "development":
            object.__setattr__(self, "API_WORKERS", 1)
        elif self.ENVIRONMENT == "production" and self.API_WORKERS < 2:
            object.__setattr__(self, "API_WORKERS", 2)

        # Chemins adaptatifs
        if self.KUBERNETES_NAMESPACE:
            data_dir = Path("/data")
            log_dir = Path("/logs")
            models_dir = Path("/app/api/models")
        elif os.path.exists("/.dockerenv"):
            data_dir = Path("/app/data")
            log_dir = Path("/app/logs")
            models_dir = Path("/app/api/models")
        elif os.getenv("GITHUB_ACTIONS"):
            data_dir = Path("/tmp/data")
            log_dir = Path("/tmp/logs")
            models_dir = self.BASE_DIR / "app/api/models"
        else:
            data_dir = self.BASE_DIR / "data"
            log_dir = self.BASE_DIR / "logs"
            models_dir = self.BASE_DIR / "app/api/models"

        object.__setattr__(self, "DATA_DIR", data_dir)
        object.__setattr__(self, "LOG_DIR", log_dir)
        object.__setattr__(self, "MODELS_DIR", models_dir)

        for d in (data_dir, log_dir, models_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                pass

        if self.ENVIRONMENT == "production" and (
            self.API_SECRET_KEY == (
                "development-secret-key-change-in-production"
            )
        ):
            raise ValueError("API_SECRET_KEY doit être changée en production")

        return self

    # -------- Helpers --------
    @property
    def allowed_origins_list(self) -> List[str]:
        return (
            ["*"]
            if not self.ALLOWED_ORIGINS
            else [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]
        )

    @property
    def cors_origins_list(self) -> List[str]:
        """
        Retourne une liste des origines autorisées pour CORS.
        """
        return (
            ["*"]
            if not self.CORS_ORIGINS
            else [o.strip() for o in self.CORS_ORIGINS.split(",")]
        )

    @property
    def trusted_hosts_list(self) -> List[str]:
        return (
            ["*"]
            if self.TRUSTED_HOSTS == "*"
            else [h.strip() for h in self.TRUSTED_HOSTS.split(",")]
        )

    def get_model_path(self, model_name: str) -> Path:
        return self.MODELS_DIR / f"{model_name}.joblib"

    def get_data_path(self, filename: str) -> Path:
        return self.DATA_DIR / filename

    def get_log_path(self, filename: str = "app.log") -> Path:
        return self.LOG_DIR / filename


settings = Settings()
