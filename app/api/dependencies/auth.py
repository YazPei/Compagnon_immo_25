# app/api/dependencies/auth.py
from typing import Annotated

from fastapi import Header, HTTPException, status
from app.api.config.settings import settings

TEST_KEY = "test_api_key"

def _is_valid(key: str) -> bool:
    return key in {TEST_KEY, settings.API_SECRET_KEY}

async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
) -> str:
    """401 si manquant ou invalide (utilisé par /api/v1 et /api/v1/estimation)."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing",
        )
    if not _is_valid(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key

async def verify_api_key_required(
    x_api_key: Annotated[str, Header(alias="X-API-Key")]
) -> str:
    """422 si manquant, sinon 401 si invalide (utilisé par /api/v1/historique)."""
    if not _is_valid(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key

