import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DVCConnector:
    """Connecteur pour gérer les interactions avec DVC."""

    def __init__(self, repo_path: Optional[str] = None):
        """Initialise le connecteur DVC avec détection d'environnement."""
        # Variables d'environnement pour DagsHub
        self.dagshub_username = os.getenv("DAGSHUB_USERNAME")
        self.dagshub_token = os.getenv("DAGSHUB_TOKEN")

        # Détection automatique du chemin selon l'environnement
        if repo_path:
            self.repo_path = Path(repo_path)
        elif os.getenv("KUBERNETES_NAMESPACE"):
            self.repo_path = Path("/app")
        elif os.path.exists("/.dockerenv"):
            self.repo_path = Path("/app")
        elif os.getenv("GITHUB_ACTIONS"):
            self.repo_path = Path("/github/workspace")
        else:
            # Développement local - détection automatique
            current_path = Path.cwd()
            if "Compagnon_immo" in str(current_path):
                self.repo_path = current_path
            else:
                self.repo_path = Path("/home/ketsiapedro/Bureau/MLE/Compagnon_immo")

        self.models_dir = self.repo_path / "app/api/models"
        self.dvc_file = self.repo_path / "dvc.yaml"

        logger.info(f"📂 Chemin DVC: {self.repo_path}")
        logger.info(f"🤖 Répertoire modèles: {self.models_dir}")

        # Vérification de l'existence des fichiers critiques
        if not self.dvc_file.exists():
            logger.warning(f"⚠️ Fichier DVC non trouvé : {self.dvc_file}")

    def is_dvc_available(self) -> bool:
        """Vérifie si DVC est installé et accessible."""
        try:
            result = subprocess.run(
                ["dvc", "--version"], capture_output=True, text=True, timeout=10
            )
            logger.info(f"DVC version: {result.stdout.strip()}")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"❌ DVC non disponible : {e}")
            return False

    def get_model_files(self) -> Dict[str, str]:
        """Récupère les fichiers de modèles disponibles dans le répertoire DVC."""
        model_files = {}
        try:
            if self.models_dir.exists():
                for pattern in ["*.joblib", "*.pkl"]:
                    for file_path in self.models_dir.glob(pattern):
                        name = file_path.stem
                        model_files[name] = str(file_path)
                logger.info(f"✅ Modèles trouvés : {list(model_files.keys())}")
            else:
                logger.warning(
                    f"⚠️ Répertoire des modèles introuvable : {self.models_dir}"
                )
        except Exception as e:
            logger.error(
                f"❌ Erreur lors de la récupération des fichiers de modèles : {e}"
            )
        return model_files

    def pull_latest_models(self) -> Dict[str, Any]:
        """Récupère les derniers modèles depuis DVC."""
        try:
            logger.info("🔄 Début de la synchronisation des modèles avec DVC...")
            result = subprocess.run(
                ["dvc", "pull"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                logger.info("✅ Modèles synchronisés avec succès.")
                return {
                    "status": "success",
                    "message": "Modèles mis à jour avec succès.",
                    "models": self.get_model_files(),
                }
            else:
                logger.warning(
                    f"⚠️ Problème lors de la synchronisation DVC : {result.stderr.strip()}"
                )
                return {
                    "status": "warning",
                    "message": result.stderr.strip(),
                    "models": self.get_model_files(),
                }
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout : La commande DVC pull a pris trop de temps.")
            return {
                "status": "timeout",
                "message": "La synchronisation DVC a dépassé le délai imparti (2 minutes).",
            }
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synchronisation DVC : {e}")
            return {"status": "error", "message": str(e)}

    def check_models_status(self) -> Dict[str, Any]:
        """Vérifie l'état des modèles dans DVC."""
        try:
            logger.info("🔄 Vérification de l'état des modèles avec DVC...")
            result = subprocess.run(
                ["dvc", "status"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            models = self.get_model_files()
            needs_update = "modified" in result.stdout.lower()

            if result.returncode == 0:
                logger.info("✅ État des modèles récupéré avec succès.")
            else:
                logger.warning(
                    f"⚠️ Problème lors de la vérification de l'état DVC : {result.stderr.strip()}"
                )

            return {
                "dvc_available": self.is_dvc_available(),
                "models_count": len(models),
                "models": models,
                "dvc_status": (
                    result.stdout.strip() if result.returncode == 0 else "Erreur"
                ),
                "needs_update": needs_update,
            }
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout : La commande DVC status a pris trop de temps.")
            return {
                "status": "timeout",
                "message": "La vérification de l'état DVC a dépassé le délai imparti (30 secondes).",
                "models": self.get_model_files(),
            }
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification de l'état DVC : {e}")
            return {
                "status": "error",
                "message": str(e),
                "models": self.get_model_files(),
            }

    # app/api/services/dvc_connector.py


"""
Connecteur DVC minimal pour les tests.
Expose dvc_service() qui retourne un objet avec les méthodes attendues (no-op).
"""


class _NoopDVC:
    def __init__(self): ...
    def ensure_data(self): ...
    def pull(self, *args, **kwargs): ...
    def status(self, *args, **kwargs):
        return {}


def dvc_service():
    return _NoopDVC()


# Instance globale du connecteur DVC
dvc_connector = DVCConnector()
