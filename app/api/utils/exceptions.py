"""
Exceptions personnalisées pour l'API Compagnon Immobilier.
"""

from fastapi import HTTPException
from typing import Any, Dict, Optional


class CompagnionImmoException(HTTPException):
    """Exception de base pour l'application Compagnon Immo."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code


class EstimationError(CompagnionImmoException):
    """Erreurs liées à l'estimation de prix."""

    def __init__(self, detail: str, error_code: str = "ESTIMATION_ERROR"):
        super().__init__(
            status_code=400,
            detail=f"Erreur d'estimation: {detail}",
            error_code=error_code,
        )


class ModelNotFoundError(CompagnionImmoException):
    """Erreur quand un modèle n'est pas trouvé."""

    def __init__(self, model_name: str):
        super().__init__(
            status_code=404,
            detail=f"Modèle non trouvé: {model_name}",
            error_code="MODEL_NOT_FOUND",
        )


class ModelLoadingError(CompagnionImmoException):
    """Erreur lors du chargement d'un modèle."""

    def __init__(self, model_name: str, detail: str):
        super().__init__(
            status_code=500,
            detail=f"Erreur de chargement du modèle {model_name}: {detail}",
            error_code="MODEL_LOADING_ERROR",
        )


class DVCError(CompagnionImmoException):
    """Erreurs liées à DVC."""

    def __init__(self, detail: str, error_code: str = "DVC_ERROR"):
        super().__init__(
            status_code=500,
            detail=f"Erreur DVC: {detail}",
            error_code=error_code,
        )


class DatabaseError(CompagnionImmoException):
    """Erreurs liées à la base de données."""

    def __init__(self, detail: str, error_code: str = "DATABASE_ERROR"):
        super().__init__(
            status_code=500,
            detail=f"Erreur base de données: {detail}",
            error_code=error_code,
        )


class ValidationError(CompagnionImmoException):
    """Erreurs de validation des données."""

    def __init__(self, detail: str, field: Optional[str] = None):
        error_msg = f"Erreur de validation"
        if field:
            error_msg += f" pour le champ '{field}'"
        error_msg += f": {detail}"

        super().__init__(
            status_code=422,
            detail=error_msg,
            error_code="VALIDATION_ERROR",
        )


class AuthenticationError(CompagnionImmoException):
    """Erreurs d'authentification."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=401,
            detail=detail,
            error_code="AUTHENTICATION_ERROR",
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(CompagnionImmoException):
    """Erreurs d'autorisation."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=403,
            detail=detail,
            error_code="AUTHORIZATION_ERROR",
        )


class RateLimitError(CompagnionImmoException):
    """Erreur de limitation du taux."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=429,
            detail=f"Trop de requêtes. Réessayez dans {retry_after} secondes.",
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": str(retry_after)},
        )


class ExternalServiceError(CompagnionImmoException):
    """Erreur avec un service externe."""

    def __init__(self, service_name: str, detail: str):
        super().__init__(
            status_code=503,
            detail=f"Service {service_name} indisponible: {detail}",
            error_code="EXTERNAL_SERVICE_ERROR",
        )


class ConfigurationError(CompagnionImmoException):
    """Erreur de configuration."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            detail=f"Erreur de configuration: {detail}",
            error_code="CONFIGURATION_ERROR",
        )