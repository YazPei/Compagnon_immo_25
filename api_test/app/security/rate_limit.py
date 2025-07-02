from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from api_test.app.cache.redis_cache import get_redis_client
import time

RATE_LIMIT_HOURLY = 100
RATE_LIMIT_DAILY = 1000

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        redis = get_redis_client()
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return await call_next(request)
        now = int(time.time())
        hour = now // 3600
        day = now // 86400
        hour_key = f"rl:{api_key}:h:{hour}"
        day_key = f"rl:{api_key}:d:{day}"
        # Incrémentation atomique
        h_count = redis.incr(hour_key)
        d_count = redis.incr(day_key)
        if h_count == 1:
            redis.expire(hour_key, 3600)
        if d_count == 1:
            redis.expire(day_key, 86400)
        if h_count > RATE_LIMIT_HOURLY:
            raise HTTPException(status_code=429, detail={"error": "Limite de requêtes dépassée", "retry_after": 3600})
        if d_count > RATE_LIMIT_DAILY:
            raise HTTPException(status_code=429, detail={"error": "Limite de requêtes dépassée", "retry_after": 86400})
        return await call_next(request) 