"""
Middleware pour capturer et gérer les erreurs dans l'application.
"""

import logging
from typing import Awaitable, Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("error_middleware")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            response: Response = await call_next(request)
            return response
        except HTTPException as http_exc:
            logger.error(f"HTTPException: {http_exc.detail}")
            return JSONResponse(
                status_code=http_exc.status_code,
                content={"detail": http_exc.detail},
            )
        except Exception:
            logger.exception("Une erreur inattendue s'est produite.")
            return JSONResponse(
                status_code=500,
                content={"detail": "Une erreur interne s'est produite."},
            )
