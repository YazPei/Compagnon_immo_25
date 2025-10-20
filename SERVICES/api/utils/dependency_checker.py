"""
Module pour centraliser la vérification des dépendances.
"""

import logging

import httpx
import redis
from sqlalchemy import text

from app.api.db.database import engine

logger = logging.getLogger(__name__)


async def check_database():
    """Vérifie la santé de la base de données."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        return {"status": "healthy", "message": "Base de données accessible"}
    except Exception as e:
        logger.error(f"Erreur base de données : {str(e)}")
        return {"status": "unhealthy", "message": f"Erreur base de données : {str(e)}"}


async def check_redis(redis_url):
    """Vérifie la connexion à Redis."""
    try:
        r = redis.from_url(redis_url, decode_responses=True)
        r.ping()
        return {"status": "healthy", "message": "Redis accessible"}
    except Exception as e:
        logger.error(f"Erreur Redis : {str(e)}")
        return {"status": "unhealthy", "message": f"Redis inaccessible : {str(e)}"}


async def check_mlflow(mlflow_url):
    """Vérifie la connexion à MLflow."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{mlflow_url}/health")
            if response.status_code == 200:
                return {"status": "healthy", "message": "MLflow accessible"}
            return {
                "status": "degraded",
                "message": f"MLflow status: {response.status_code}",
            }
    except Exception as e:
        logger.error(f"Erreur MLflow : {str(e)}")
        return {"status": "unhealthy", "message": f"MLflow inaccessible : {str(e)}"}
