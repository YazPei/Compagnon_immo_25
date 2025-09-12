from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from collections import defaultdict
from typing import Dict, List
import time
import logging

from app.api.config.settings import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Gestionnaire de limitation du taux de requêtes."""

    def __init__(self, max_requests: int = None, window_seconds: int = None):
        self.max_requests = max_requests or settings.RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or settings.RATE_LIMIT_WINDOW
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Vérifie si une requête est autorisée."""
        now = time.time()

        # Nettoyer les anciennes requêtes hors de la fenêtre
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window_seconds
        ]

        # Vérifier si la limite est atteinte
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"⚠️ Rate limit dépassé pour {client_ip}")
            return False

        # Ajouter la requête actuelle
        self.requests[client_ip].append(now)
        return True

    def get_reset_time(self, client_ip: str) -> int:
        """Retourne le temps restant avant le reset de la fenêtre."""
        if not self.requests[client_ip]:
            return 0

        oldest_request = min(self.requests[client_ip])
        return int(self.window_seconds - (time.time() - oldest_request))


# Instance globale du gestionnaire de limitation
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """Middleware de limitation du taux de requêtes."""
    client_ip = request.client.host

    # Exempter certains endpoints de la limitation
    exempted_paths = ["/health", "/", "/docs", "/openapi.json"]
    if request.url.path in exempted_paths:
        return await call_next(request)

    # Vérifier si le client est autorisé
    if not rate_limiter.is_allowed(client_ip):
        reset_time = rate_limiter.get_reset_time(client_ip)
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too many requests",
                "detail": f"Rate limit exceeded. Try again in {reset_time} seconds.",
                "retry_after": reset_time
            },
            headers={"Retry-After": str(reset_time)}
        )

    # Continuer la requête
    response = await call_next(request)

    # Ajouter des headers informatifs sur la limitation
    remaining = rate_limiter.max_requests - len(rate_limiter.requests[client_ip])
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(time.time()) + rate_limiter.window_seconds)

    return response