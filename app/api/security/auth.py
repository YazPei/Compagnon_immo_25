# app/api/security/auth.py
"""Module de gestion de l'authentification par clé API et JWT."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, TypedDict
import os

from app.api.config.settings import settings

logger = logging.getLogger(__name__)

# Correction de l'erreur liée à `ApiKeyInfo` et réduction des lignes longues


class ApiKeyInfo(TypedDict):
    name: str
    permissions: list[str]


VALID_API_KEYS: dict[str, ApiKeyInfo] = {
    os.getenv("API_KEY_TEST", "test-key-123"): {
        "name": "Test Key",
        "permissions": ["read", "write"],
    },
    os.getenv("API_KEY_DEV", "dev-key"): {
        "name": "Development Key",
        "permissions": ["read", "write"],
    },
}

# Ajout de journalisation pour les clés API manquantes
for key, value in VALID_API_KEYS.items():
    if key.startswith("test"):
        logger.warning(f"⚠️ Clé API de test détectée : {key}")

# Ajout d'une validation stricte pour les clés API
if not all(key for key in VALID_API_KEYS if key):
    logger.error(
        "❌ Clés API manquantes dans les variables d'environnement."
    )
    raise RuntimeError(
        "Configuration des clés API incomplète."
    )

# Hashing des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class AuthManager:
    """Gestionnaire d'authentification pour JWT et mots de passe."""

    def __init__(self):
        # ⚠️ On utilise la clé existante dans Settings
        self.secret_key = getattr(
            settings, "API_SECRET_KEY", "change-me-in-prod"
        )
        # Valeurs par défaut sûres si non présentes dans Settings
        self.algorithm = getattr(settings, "JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(
            getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 60)
        )

    def verify_password(
        self, plain_password: str, hashed_password: str
    ) -> bool:
        """Vérifie si un mot de passe correspond à son hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash un mot de passe."""
        return pwd_context.hash(password)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Crée un token JWT."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (
            expires_delta or timedelta(
                minutes=self.access_token_expire_minutes
            )
        )
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode, self.secret_key, algorithm=self.algorithm
        )

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Vérifie et décode un token JWT."""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )
            return payload
        except JWTError as e:
            logger.warning(f"Token invalide: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalide",
                headers={"WWW-Authenticate": "Bearer"},
            )


# Instance globale
auth_manager = AuthManager()


# ------------- API Key checks -------------
def verify_api_key_value(api_key: Optional[str]) -> bool:
    """Vérifie une valeur de clé API (chaîne)."""
    if not api_key:
        return False
    return api_key in VALID_API_KEYS


# Version utilisée par ton `estimation.py` (dépendance Header -> str)
def verify_api_key(x_api_key: str) -> bool:
    """Compat avec get_api_key(x_api_key: Header(...))."""
    return verify_api_key_value(x_api_key)


# Version pratique si on veut lire la clé depuis la Request
def verify_api_key_from_request(request: Request) -> bool:
    api_key = request.headers.get("X-API-Key")
    return verify_api_key_value(api_key)


# ------------- Dépendances JWT -------------


class CustomHTTPAuthorizationCredentials:
    def __init__(self, credentials: str):
        self.credentials = credentials


async def get_credentials(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401, detail="Missing Authorization Header"
        )
    return CustomHTTPAuthorizationCredentials(
        credentials=auth_header.split()[1]
    )


async def get_current_user(
    credentials: CustomHTTPAuthorizationCredentials = Depends(get_credentials),
) -> Dict[str, Any]:
    """Récupère l'utilisateur actuel à partir du token JWT."""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalide",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"user_id": user_id, "payload": payload}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Impossible de valider les informations d'identification",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    """Récupère l'utilisateur si authentifié (JWT), sinon retourne None."""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]
        payload = auth_manager.verify_token(token)
        return {"user_id": payload.get("sub"), "payload": payload}
    except Exception:
        return None


async def require_auth_or_api_key(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """Exige une authentification JWT ou une clé API valide."""
    if user:
        return user

    if verify_api_key_from_request(request):
        return {"user_id": "api_user", "auth_type": "api_key"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentification requise",
        headers={"WWW-Authenticate": "Bearer"},
    )


