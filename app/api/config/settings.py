# app/api/config/settings.py
"""
Configuration centralisée (Pydantic v2 + pydantic-settings).
Inclut alias JWT, rate limiting et chemins adaptatifs (Docker/K8S/CI/local).
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
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
    PROJECT_NAME: str = Field(default="Compagnon Immobilier", env="PROJECT_NAME")
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    VERSION: str = Field(default="1.0.0", env="VERSION")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # --- API ---
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=1, env="API_WORKERS")
    API_V1_STR: str = Field(default="/api/v1", env="API_V1_STR")

    # --- Sécurité ---
    # IMPORTANT: Les tests attendent X-API-Key == "test_api_key"
    API_KEY: str = Field(default="test_api_key", env="API_KEY")

    # --- DB ---
    DATABASE_URL: str = Field(default="sqlite:///./app.db", env="DATABASE_URL")
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")

    # --- Redis ---
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_SSL: bool = Field(default=False, env="REDIS_SSL")

    # --- MLflow ---
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = Field(default=None, env="MLFLOW_S3_ENDPOINT_URL")

    # --- DVC / DagsHub ---
    DVC_REMOTE: str = Field(default="origin", env="DVC_REMOTE")
    DAGSHUB_URL: str = Field(default="https://dagshub.com/YazPei/compagnon_immo.dvc", env="DAGSHUB_URL")
    DAGSHUB_USERNAME: Optional[str] = Field(default=None, env="DAGSHUB_USERNAME")
    DAGSHUB_PASSWORD: Optional[str] = Field(default=None, env="DAGSHUB_PASSWORD")

    # --- JWT ---
    API_SECRET_KEY: str = Field(
        default="development-secret-key-change-in-production",
        env="API_SECRET_KEY",
        min_length=32,
    )
    JWT_SECRET_KEY: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # --- Rate limiting ---
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # en secondes
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, env="RATE_LIMIT_WINDOW_SECONDS")  # alias

    # --- CORS / hosts ---
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8501,http://127.0.0.1:3000",
        env="CORS_ORIGINS",
    )
    ALLOWED_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8501", env="ALLOWED_ORIGINS")
    TRUSTED_HOSTS: str = Field(default="*", env="TRUSTED_HOSTS")

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # --- Logs ---
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")

    # --- Monitoring ---
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    METRICS_PATH: str = Field(default="/metrics", env="METRICS_PATH")

    # --- Kubernetes ---
    KUBERNETES_NAMESPACE: Optional[str] = Field(default=None, env="KUBERNETES_NAMESPACE")
    KUBERNETES_SERVICE_NAME: Optional[str] = Field(default=None, env="KUBERNETES_SERVICE_NAME")

    # --- Santé ---
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    READINESS_TIMEOUT: int = Field(default=5, env="READINESS_TIMEOUT")
    LIVENESS_TIMEOUT: int = Field(default=5, env="LIVENESS_TIMEOUT")

    # --- Airflow ---
    AIRFLOW_HOME: Optional[str] = Field(default=None, env="AIRFLOW_HOME")
    AIRFLOW_CONN_ID: str = Field(default="compagnon_immo_api", env="AIRFLOW_CONN_ID")

    # -------- Validators / Hooks --------
    @field_validator("ENVIRONMENT")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT doit être dans {sorted(allowed)}")
        return v

    @model_validator(mode="after")
    def _after(self) -> "Settings":
        # alias JWT: si non fourni, reprendre API_SECRET_KEY
        if not self.JWT_SECRET_KEY:
            object.__setattr__(self, "JWT_SECRET_KEY", self.API_SECRET_KEY)

        # assurer cohérence des deux noms de fenêtre RL
        if self.RATE_LIMIT_WINDOW <= 0 and self.RATE_LIMIT_WINDOW_SECONDS > 0:
            object.__setattr__(self, "RATE_LIMIT_WINDOW", self.RATE_LIMIT_WINDOW_SECONDS)
        elif self.RATE_LIMIT_WINDOW > 0 and self.RATE_LIMIT_WINDOW_SECONDS <= 0:
            object.__setattr__(self, "RATE_LIMIT_WINDOW_SECONDS", self.RATE_LIMIT_WINDOW)

        # Ajustement workers
        if self.ENVIRONMENT == "development":
            object.__setattr__(self, "API_WORKERS", 1)
        elif self.ENVIRONMENT == "production" and self.API_WORKERS < 2:
            object.__setattr__(self, "API_WORKERS", 2)

        # Chemins adaptatifs
        if self.KUBERNETES_NAMESPACE:
            data_dir = Path("/data"); log_dir = Path("/logs"); models_dir = Path("/app/api/models")
        elif os.path.exists("/.dockerenv"):
            data_dir = Path("/app/data"); log_dir = Path("/app/logs"); models_dir = Path("/app/api/models")
        elif os.getenv("GITHUB_ACTIONS"):
            data_dir = Path("/tmp/data"); log_dir = Path("/tmp/logs"); models_dir = self.BASE_DIR / "app/api/models"
        else:
            data_dir = self.BASE_DIR / "data"; log_dir = self.BASE_DIR / "logs"; models_dir = self.BASE_DIR / "app/api/models"

        object.__setattr__(self, "DATA_DIR", data_dir)
        object.__setattr__(self, "LOG_DIR", log_dir)
        object.__setattr__(self, "MODELS_DIR", models_dir)

        for d in (data_dir, log_dir, models_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                pass

        if self.ENVIRONMENT == "production" and (
            self.API_SECRET_KEY == "development-secret-key-change-in-production"
        ):
            raise ValueError("API_SECRET_KEY doit être changée en production")

        return self

    # -------- Helpers --------
    @property
    def allowed_origins_list(self) -> List[str]:
        return ["*"] if not self.ALLOWED_ORIGINS else [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        return ["*"] if not self.CORS_ORIGINS else [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def trusted_hosts_list(self) -> List[str]:
        return ["*"] if self.TRUSTED_HOSTS == "*" else [h.strip() for h in self.TRUSTED_HOSTS.split(",")]

    def get_model_path(self, model_name: str) -> Path:
        return self.MODELS_DIR / f"{model_name}.joblib"

    def get_data_path(self, filename: str) -> Path:
        return self.DATA_DIR / filename

    def get_log_path(self, filename: str = "app.log") -> Path:
        return self.LOG_DIR / filename


settings = Settings()

