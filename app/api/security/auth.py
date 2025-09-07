"""Module de gestion de l'authentification par clé API et JWT."""

from fastapi import Depends, HTTPException, Header, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from app.api.config.settings import settings

logger = logging.getLogger(__name__)

# Configuration des clés API valides
VALID_API_KEYS = {
    "test-key-123": {"name": "Test Key", "permissions": ["read", "write"]},
    "test_api_key": {"name": "Test API Key", "permissions": ["read", "write"]},
    "dev-key": {"name": "Development Key", "permissions": ["read", "write"]},
}

# Configuration pour le hashing des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class AuthManager:
    """Gestionnaire d'authentification pour JWT et mots de passe."""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Vérifie si un mot de passe correspond à son hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash un mot de passe."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Crée un token JWT."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Vérifie et décode un token JWT."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"Token invalide: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalide",
                headers={"WWW-Authenticate": "Bearer"},
            )


# Instance globale du gestionnaire d'authentification
auth_manager = AuthManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
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


def verify_api_key(request: Request) -> bool:
    """Vérifie si une clé API est valide."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return False
    return api_key in VALID_API_KEYS


async def require_auth_or_api_key(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> Dict[str, Any]:
    """Exige une authentification JWT ou une clé API valide."""
    if user:
        return user
    
    if verify_api_key(request):
        return {"user_id": "api_user", "auth_type": "api_key"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentification requise",
        headers={"WWW-Authenticate": "Bearer"},
    )