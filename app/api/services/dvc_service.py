import logging
from typing import Dict, Any
from .dvc_connector import dvc_connector

logger = logging.getLogger(__name__)

class DVCService:
    """Service DVC pour la gestion des modèles."""
    
    def __init__(self):
        self.connector = dvc_connector
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """État complet de DVC."""
        try:
            return {
                "environment": {
                    "dvc_available": self.connector.is_dvc_available(),
                    "dvc_repo": self.connector.dvc_file.exists()
                },
                "paths": {
                    "dvc_dir": str(self.connector.repo_path),
                    "models_dir": str(self.connector.models_dir)
                },
                "models": self.connector.get_model_files()
            }
        except Exception as e:
            logger.error(f"Erreur status DVC: {e}")
            return {
                "environment": {"dvc_available": False, "dvc_repo": False},
                "paths": {},
                "models": {}
            }
    
    def sync_models(self) -> Dict[str, Any]:
        """Synchronise les modèles."""
        return self.connector.pull_latest_models()

# Instance globale
dvc_service = DVCService()