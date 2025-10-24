import asyncio
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import psutil
import requests

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    details: Dict[str, Any] = None

class HealthChecker:
    """Système de vérification de santé complet."""
    
    def __init__(self):
        self.checks = []
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Exécuter toutes les vérifications de santé."""
        results = []
        
        # Vérifications système
        results.append(await self._check_memory())
        results.append(await self._check_cpu())
        results.append(await self._check_disk())
        
        # Vérifications ML
        results.append(await self._check_model_availability())
        results.append(await self._check_data_freshness())
        
        # Vérifications externes
        results.append(await self._check_external_services())
        
        # Calculer le statut global
        overall_status = self._calculate_overall_status(results)
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details or {}
                }
                for check in results
            ]
        }
    
    async def _check_memory(self) -> HealthCheck:
        """Vérifier l'utilisation mémoire."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {usage_percent:.1f}%"
            elif usage_percent > 80:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent:.1f}%"
            
            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    "usage_percent": usage_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory: {str(e)}",
                timestamp=time.time()
            )
    
    async def _check_cpu(self) -> HealthCheck:
        """Vérifier l'utilisation CPU."""
        try:
            # Moyenne sur 1 seconde
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheck(
                name="cpu",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    "usage_percent": cpu_percent,
                    "core_count": psutil.cpu_count()
                }
            )
        except Exception as e:
            return HealthCheck(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check CPU: {str(e)}",
                timestamp=time.time()
            )
    
    async def _check_disk(self) -> HealthCheck:
        """Vérifier l'espace disque."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {usage_percent:.1f}%"
            elif usage_percent > 80:
                status = HealthStatus.WARNING
                message = f"Disk usage high: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}%"
            
            return HealthCheck(
                name="disk",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    "usage_percent": usage_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                }
            )
        except Exception as e:
            return HealthCheck(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk: {str(e)}",
                timestamp=time.time()
            )
    
    async def _check_model_availability(self) -> HealthCheck:
        """Vérifier la disponibilité des modèles."""
        try:
            # Vérifier que les fichiers de modèle existent
            import os
            model_path = "outputs/models"
            
            if not os.path.exists(model_path):
                return HealthCheck(
                    name="model_availability",
                    status=HealthStatus.CRITICAL,
                    message="Model directory not found",
                    timestamp=time.time()
                )
            
            model_files = os.listdir(model_path)
            model_count = len([f for f in model_files if f.endswith('.h5')])
            
            if model_count == 0:
                status = HealthStatus.CRITICAL
                message = "No trained models found"
            else:
                status = HealthStatus.HEALTHY
                message = f"{model_count} models available"
            
            return HealthCheck(
                name="model_availability",
                status=status,
                message=message,
                timestamp=time.time(),
                details={"model_count": model_count}
            )
        except Exception as e:
            return HealthCheck(
                name="model_availability",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check models: {str(e)}",
                timestamp=time.time()
            )
    
    async def _check_data_freshness(self) -> HealthCheck:
        """Vérifier la fraîcheur des données."""
        try:
            import os
            data_path = "data"
            
            if not os.path.exists(data_path):
                return HealthCheck(
                    name="data_freshness",
                    status=HealthStatus.WARNING,
                    message="Data directory not found",
                    timestamp=time.time()
                )
            
            # Trouver le fichier le plus récent
            latest_time = 0
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    mtime = os.path.getmtime(file_path)
                    latest_time = max(latest_time, mtime)
            
            if latest_time == 0:
                return HealthCheck(
                    name="data_freshness",
                    status=HealthStatus.WARNING,
                    message="No data files found",
                    timestamp=time.time()
                )
            
            hours_old = (time.time() - latest_time) / 3600
            
            if hours_old > 24:
                status = HealthStatus.WARNING
                message = f"Data is {hours_old:.1f} hours old"
            else:
                status = HealthStatus.HEALTHY
                message = f"Data is {hours_old:.1f} hours old"
            
            return HealthCheck(
                name="data_freshness",
                status=status,
                message=message,
                timestamp=time.time(),
                details={"hours_old": hours_old}
            )
        except Exception as e:
            return HealthCheck(
                name="data_freshness",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check data freshness: {str(e)}",
                timestamp=time.time()
            )
    
    async def _check_external_services(self) -> HealthCheck:
        """Vérifier les services externes."""
        try:
            # Test de connectivité réseau
            response = requests.get("https://httpbin.org/status/200", timeout=5)
            
            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = "External connectivity OK"
            else:
                status = HealthStatus.WARNING
                message = f"External service returned {response.status_code}"
            
            return HealthCheck(
                name="external_services",
                status=status,
                message=message,
                timestamp=time.time()
            )
        except Exception as e:
            return HealthCheck(
                name="external_services",
                status=HealthStatus.WARNING,
                message=f"External connectivity failed: {str(e)}",
                timestamp=time.time()
            )
    
    def _calculate_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Calculer le statut global."""
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        elif any(check.status == HealthStatus.UNKNOWN for check in checks):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

# Instance globale
health_checker = HealthChecker()
