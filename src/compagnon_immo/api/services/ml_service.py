"""
Service ML utilisant les métriques centralisées.
"""

import asyncio
import glob
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from compagnon_immo.api.monitoring.prometheus_registry import (MODEL_LOAD_TIME,
                                                    MODELS_LOADED,
                                                    metrics_collector)
from compagnon_immo.api.services.dvc_connector import dvc_service

logger = logging.getLogger(__name__)


class MLService:
    """Service de Machine Learning avec métriques intégrées."""

    def __init__(self):
        self.models = {}
        self.model_versions = {}
        self.preprocessors = {}
        self.models_path = Path(__file__).parent.parent / "models"
        self.is_loaded = False
        self.last_sync = None
        self.load_errors = []

    async def load_models_from_dvc(self) -> Dict[str, Any]:
        """Charger tous les modèles disponibles via DVC."""
        load_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "models_loaded": 0,
            "preprocessors_loaded": 0,
            "errors": [],
            "dvc_sync": None,
        }

        try:
            logger.info("🤖 Chargement des modèles ML avec DVC...")

            sync_result = dvc_service.sync_models()
            load_result["dvc_sync"] = sync_result

            available_models = sync_result.get("models", {})

            if not available_models:
                logger.warning("⚠️ Aucun modèle trouvé")
                load_result["status"] = "no_models"
                return load_result

            for model_key, model_info in available_models.items():
                try:
                    model_path = Path(model_info["path"])

                    if model_path.exists():
                        if self._is_preprocessor(model_key, model_path):
                            preprocessor = self._load_artifact(model_path)
                            self.preprocessors[model_key] = preprocessor
                            load_result["preprocessors_loaded"] += 1
                            logger.info(f"✅ Préprocesseur chargé: {model_key}")
                        else:
                            model = self._load_artifact(model_path)
                            self.models[model_key] = model
                            load_result["models_loaded"] += 1
                            logger.info(f"✅ Modèle chargé: {model_key}")
                    else:
                        error_msg = f"Fichier non trouvé: {model_path}"
                        load_result["errors"].append(error_msg)
                        logger.warning(f"⚠️ {error_msg}")

                except Exception as e:
                    error_msg = f"Erreur chargement {model_key}: {e}"
                    load_result["errors"].append(error_msg)
                    logger.error(f"❌ {error_msg}")

            if self.models or self.preprocessors:
                self.is_loaded = True
                self.last_sync = datetime.utcnow()
                load_result["status"] = "success"
                logger.info(
                    f"✅ Chargement terminé: {len(self.models)} modèles, {len(self.preprocessors)} préprocesseurs"
                )
            else:
                load_result["status"] = "failed"
                logger.error("❌ Aucun modèle n'a pu être chargé")

            return load_result

        except Exception as e:
            error_msg = f"Erreur critique lors du chargement: {e}"
            logger.error(error_msg)
            load_result["status"] = "error"
            load_result["errors"].append(error_msg)
            return load_result

    def _load_artifact(self, file_path: Path):
        """Charger un artifact (modèle ou préprocesseur)."""
        try:
            if file_path.suffix == ".joblib":
                return joblib.load(file_path)
            elif file_path.suffix == ".pkl":
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Erreur chargement {file_path}: {e}")
            raise

    def get_model(self, name: str) -> Any:
        """Retourne le modèle par nom, ou None."""
        return self.models.get(name)

    def get_model_metadata(self, name: str) -> Dict[str, Any]:
        """Retourne les métadonnées du modèle par nom, ou vide."""
        return self.metadata.get(name, {})

    def _is_preprocessor(self, model_key: str, model_path: Path) -> bool:
        """Déterminer si c'est un préprocesseur."""
        preprocessor_keywords = [
            "preprocessor",
            "scaler",
            "encoder",
            "transformer",
            "feature",
            "prep",
            "preprocessing",
        ]

        key_lower = model_key.lower()
        path_lower = str(model_path).lower()

        return any(
            keyword in key_lower or keyword in path_lower
            for keyword in preprocessor_keywords
        )

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Effectuer une prédiction avec les modèles chargés."""
        if not self.is_loaded:
            raise RuntimeError(
                "Aucun modèle chargé. Appelez load_models_from_dvc() d'abord."
            )

        if not self.models:
            raise RuntimeError("Aucun modèle de prédiction disponible.")

        try:
            model_key = self._select_best_model()
            model = self.models[model_key]

            df = pd.DataFrame([features])

            preprocessor_key = self._find_matching_preprocessor(model_key)
            if preprocessor_key:
                preprocessor = self.preprocessors[preprocessor_key]
                df_processed = preprocessor.transform(df)
                logger.info(f"🔧 Préprocessing appliqué: {preprocessor_key}")
            else:
                df_processed = df
                logger.warning("⚠️ Aucun préprocesseur trouvé")

            if hasattr(model, "predict"):
                prediction = model.predict(df_processed)
                if isinstance(prediction, np.ndarray):
                    prediction = prediction[0] if len(prediction) > 0 else 0
            else:
                raise ValueError("Le modèle n'a pas de méthode predict")

            confidence_margin = float(prediction) * 0.15  # ±15%

            return {
                "prediction": float(prediction),
                "confidence_interval": {
                    "lower": float(prediction) - confidence_margin,
                    "upper": float(prediction) + confidence_margin,
                },
                "model_used": model_key,
                "preprocessor_used": preprocessor_key,
                "last_sync": self.last_sync.isoformat() if self.last_sync else None,
                "features_count": len(features),
            }

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise RuntimeError(f"Erreur prédiction: {e}")

    def _select_best_model(self) -> str:
        """Sélectionner le meilleur modèle disponible."""
        preferred_patterns = [
            "best_model",
            "final_model",
            "lgb",
            "lightgbm",
            "random_forest",
            "rf",
            "regression",
        ]

        for pattern in preferred_patterns:
            for model_key in self.models.keys():
                if pattern in model_key.lower():
                    return model_key

        return next(iter(self.models.keys()))

    def _find_matching_preprocessor(self, model_key: str) -> Optional[str]:
        """Trouver le préprocesseur correspondant au modèle."""
        model_base = model_key.lower().replace("models_", "")

        for prep_key in self.preprocessors.keys():
            prep_base = prep_key.lower().replace("models_", "")
            if model_base in prep_base or prep_base in model_base:
                return prep_key

        return next(iter(self.preprocessors.keys())) if self.preprocessors else None

    async def refresh_models(self) -> Dict[str, Any]:
        """Actualiser les modèles depuis DVC."""
        logger.info("🔄 Actualisation des modèles...")

        self.models.clear()
        self.preprocessors.clear()
        self.is_loaded = False

        return await self.load_models_from_dvc()

    def get_health_status(self) -> Dict[str, Any]:
        """Status de santé du service ML."""
        return {
            "service": "ml_service",
            "status": "healthy" if self.is_loaded else "not_loaded",
            "models": {"count": len(self.models), "names": list(self.models.keys())},
            "preprocessors": {
                "count": len(self.preprocessors),
                "names": list(self.preprocessors.keys()),
            },
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "load_errors": self.load_errors[-5:],  # Dernières 5 erreurs
            "dvc_integration": dvc_service.get_comprehensive_status()["environment"],
        }

    def get_models_info(self) -> Dict[str, Any]:
        """Informations détaillées sur les modèles chargés."""
        return {
            "loaded_models": {
                name: {
                    "type": str(type(model).__name__),
                    "size": f"{len(str(model))} chars",  # Approximation
                }
                for name, model in self.models.items()
            },
            "loaded_preprocessors": {
                name: {
                    "type": str(type(prep).__name__),
                    "size": f"{len(str(prep))} chars",
                }
                for name, prep in self.preprocessors.items()
            },
            "available_models": dvc_service.get_model_files(),
            "is_loaded": self.is_loaded,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
        }

    def load_model(self, model_name: str, version: str = "latest") -> bool:
        """
        Charge un modèle avec suivi des métriques.

        Args:
            model_name (str): Nom du modèle.
            version (str): Version du modèle.

        Returns:
            bool: True si chargement réussi.
        """
        start_time = time.time()

        try:
            # Simuler le chargement du modèle
            model_path = self.models_path / f"{model_name}_{version}.joblib"

            if not model_path.exists():
                logger.error(f"Modèle non trouvé: {model_path}")
                metrics_collector.record_model_prediction(
                    model_name=model_name,
                    model_version=version,
                    duration=time.time() - start_time,
                    success=False,
                )
                return False

            # Charger le modèle (simulé)
            self.models[model_name] = {
                "version": version,
                "path": str(model_path),
                "loaded_at": time.time(),
            }
            self.model_versions[model_name] = version

            # Métriques de chargement
            load_duration = time.time() - start_time
            MODEL_LOAD_TIME.labels(model_name=model_name, model_version=version).set(
                load_duration
            )

            # Mettre à jour le nombre de modèles chargés
            MODELS_LOADED.set(len(self.models))

            logger.info(
                f"Modèle {model_name} v{version} chargé en {load_duration:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
            metrics_collector.record_model_prediction(
                model_name=model_name,
                model_version=version,
                duration=time.time() - start_time,
                success=False,
            )
            return False

    def predict(
        self, model_name: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Effectue une prédiction avec suivi des métriques.

        Args:
            model_name (str): Nom du modèle.
            data (dict): Données pour la prédiction.

        Returns:
            dict: Résultat de la prédiction ou None si erreur.
        """
        start_time = time.time()

        if model_name not in self.models:
            logger.error(f"Modèle {model_name} non chargé")
            metrics_collector.record_model_prediction(
                model_name=model_name,
                model_version="unknown",
                duration=time.time() - start_time,
                success=False,
            )
            return None

        try:
            model_info = self.models[model_name]
            model_version = model_info["version"]

            # Simuler la prédiction
            prediction_result = {
                "prediction": 250000.0,  # Valeur simulée
                "confidence": 0.85,
                "model_name": model_name,
                "model_version": model_version,
            }

            # Enregistrer les métriques de succès
            metrics_collector.record_model_prediction(
                model_name=model_name,
                model_version=model_version,
                duration=time.time() - start_time,
                success=True,
            )

            return prediction_result

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction avec {model_name}: {e}")
            metrics_collector.record_model_prediction(
                model_name=model_name,
                model_version=self.model_versions.get(model_name, "unknown"),
                duration=time.time() - start_time,
                success=False,
            )
            return None

    def get_models_status(self) -> Dict[str, Any]:
        """
        Retourne le statut des modèles chargés.

        Returns:
            dict: Statut des modèles.
        """
        return {
            "models_loaded": len(self.models),
            "models": {
                name: {"version": info["version"], "loaded_at": info["loaded_at"]}
                for name, info in self.models.items()
            },
        }


# Instance globale du service ML
ml_service = MLService()
