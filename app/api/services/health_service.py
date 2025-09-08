import logging
import time
import asyncio
import psutil
import os
import platform
import sys
from typing import Dict, Any
from datetime import datetime, timedelta

from app.services.dvc_service import dvc_service
from app.services.ml_service import ml_service
from app.db.database import check_db_connection

logger = logging.getLogger(__name__)


class HealthService:
    """Service complet de health check pour l'API."""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.last_check = None
        self.check_results = {}
        self.health_status = "starting"

    async def comprehensive_check(self) -> Dict[str, Any]:
        """Vérification complète de tous les composants."""
        start_time = time.time()
        self.last_check = datetime.utcnow()

        # Résultat de base
        result = {
            "status": "healthy",
            "timestamp": self.last_check.isoformat(),
            "uptime_seconds": (self.last_check - self.start_time).total_seconds(),
            "components": {},
            "system": {}
        }

        # 1. Vérification API
        api_status = await self._check_api()
        result["components"]["api"] = api_status

        # 2. Vérification base de données
        db_status = await self._check_database()
        result["components"]["database"] = db_status

        # 3. Vérification ML service
        ml_status = await self._check_ml_service()
        result["components"]["ml_service"] = ml_status

        # 4. Vérification DVC
        dvc_status = await self._check_dvc()
        result["components"]["dvc"] = dvc_status

        # 5. Vérification système
        sys_status = self._check_system()
        result["system"] = sys_status

        # Calcul du statut global
        statuses = [
            api_status["status"],
            db_status["status"],
            ml_status["status"],
            dvc_status["status"]
        ]

        if all(s == "healthy" for s in statuses):
            result["status"] = "healthy"
        elif any(s == "critical" for s in statuses):
            result["status"] = "critical"
        else:
            result["status"] = "degraded"

        # Temps de réponse
        result["response_time_ms"] = round((time.time() - start_time) * 1000, 2)

        # Stocker le résultat
        self.check_results = result
        self.health_status = result["status"]

        return result

    async def _check_api(self) -> Dict[str, Any]:
        """Vérification de base de l'API."""
        try:
            return {
                "status": "healthy",
                "name": "Compagnon Immo API",
                "version": "1.0.0",
                "started_at": self.start_time.isoformat()
            }
        except Exception as e:
            logger.error(f"Erreur vérification API: {e}")
            return {
                "status": "critical",
                "error": str(e)
            }

    async def _check_database(self) -> Dict[str, Any]:
        """Vérification de la base de données."""
        try:
            db_ok, db_msg = check_db_connection()

            if db_ok:
                return {
                    "status": "healthy",
                    "message": "Connexion à la base de données réussie",
                    "type": "SQLite"  # À ajuster selon votre configuration
                }
            else:
                return {
                    "status": "critical",
                    "error": db_msg
                }
        except Exception as e:
            logger.error(f"Erreur vérification DB: {e}")
            return {
                "status": "critical",
                "error": str(e)
            }

    async def _check_ml_service(self) -> Dict[str, Any]:
        """Vérification du service ML."""
        try:
            ml_health = ml_service.get_health_status()

            if ml_health["status"] == "healthy" and ml_health["models"]["count"] > 0:
                return {
                    "status": "healthy",
                    "models_loaded": ml_health["models"]["count"],
                    "last_sync": ml_health["last_sync"]
                }
            elif ml_health["models"]["count"] == 0:
                return {
                    "status": "critical",
                    "error": "Aucun modèle chargé"
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Service ML partiellement fonctionnel",
                    "models_loaded": ml_health["models"]["count"]
                }
        except Exception as e:
            logger.error(f"Erreur vérification ML: {e}")
            return {
                "status": "critical",
                "error": str(e)
            }

    async def _check_dvc(self) -> Dict[str, Any]:
        """Vérification de DVC."""
        try:
            dvc_status = dvc_service.get_comprehensive_status()

            if dvc_status["environment"]["dvc_available"] and dvc_status["environment"]["dvc_repo"]:
                return {
                    "status": "healthy",
                    "available": True,
                    "repo_path": dvc_status["paths"]["dvc_dir"],
                    "models_count": len(dvc_status["models"])
                }
            elif not dvc_status["environment"]["dvc_available"]:
                return {
                    "status": "critical",
                    "error": "DVC non disponible"
                }
            else:
                return {
                    "status": "degraded",
                    "message": "DVC disponible mais repo non configuré"
                }
        except Exception as e:
            logger.error(f"Erreur vérification DVC: {e}")
            return {
                "status": "critical",
                "error": str(e)
            }

    def _check_system(self) -> Dict[str, Any]:
        """Vérification des ressources système."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total_mb": round(memory.total / (1024 * 1024), 2),
                    "used_percent": memory.percent,
                    "available_mb": round(memory.available / (1024 * 1024), 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024 * 1024 * 1024), 2),
                    "used_percent": disk.percent,
                    "free_gb": round(disk.free / (1024 * 1024 * 1024), 2)
                },
                "platform": {
                    "system": platform.system(),
                    "python": sys.version,
                    "hostname": platform.node()
                }
            }
        except Exception as e:
            logger.error(f"Erreur vérification système: {e}")
            return {
                "error": str(e)
            }


# Instance globale
health_service = HealthService()