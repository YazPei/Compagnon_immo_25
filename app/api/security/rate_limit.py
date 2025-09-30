# app/api/security/rate_limit.py
"""
Rate limiter minimal (fenêtre glissante en mémoire) + middleware FastAPI.
Pensé pour les tests : stateless par process, configurable via Settings.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Deque, Dict, Tuple

from fastapi import FastAPI, HTTPException, Request, status
from starlette.responses import Response

from app.api.config.settings import settings
from app.api.security.auth import require_auth_or_api_key

WINDOW_SECONDS: int = int(getattr(settings, "RATE_LIMIT_WINDOW", 60))
MAX_REQUESTS: int = int(getattr(settings, "RATE_LIMIT_REQUESTS", 100))

_buckets: Dict[str, Deque[datetime]] = defaultdict(deque)


def _key_from_request(req: Request) -> str:
    # Clé par IP (suffisant pour tests)
    return req.client.host if req.client else "unknown"


def rate_limiter(req: Request) -> Tuple[bool, int]:
    """
    Applique une fenêtre glissante :
    - purge les timestamps hors fenêtre
    - refuse si quota atteint, sinon enregistre et autorise
    Retourne (autorisé, restant).
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=WINDOW_SECONDS)

    key = _key_from_request(req)
    bucket = _buckets[key]

    while bucket and bucket[0] <= window_start:
        bucket.popleft()

    if len(bucket) >= MAX_REQUESTS:
        return False, 0

    bucket.append(now)
    remaining = MAX_REQUESTS - len(bucket)
    return True, remaining


def rate_limit_middleware(app: FastAPI) -> None:
    """
    Middleware à brancher sur l'app FastAPI.
    Ajoute des headers X-RateLimit-* à titre informatif.
    """

    @app.middleware("http")
    async def _rl_mw(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Vérification d'authentification
        try:
            await require_auth_or_api_key(request)
        except HTTPException as auth_exc:
            raise HTTPException(
                status_code=auth_exc.status_code,
                detail=f"Authentification échouée : {auth_exc.detail}",
            )

        allowed, remaining = rate_limiter(request)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Trop de requêtes, réessayez plus tard.",
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(MAX_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(WINDOW_SECONDS)
        return response
