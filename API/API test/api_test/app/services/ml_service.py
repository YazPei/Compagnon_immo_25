"""
Service pour gérer les modèles de machine learning.
"""
import os
from typing import Optional, Dict, Any
from api_test.app.models.ml.price_model import PriceModel
from api_test.app.models.ml.trend_model import TrendModel
import glob
from api_test.app.utils.model_loader import get_model, get_preprocessor, get_model_metadata
import pandas as pd
import datetime
import numpy as np
from api_test.app.utils.feature_enrichment import harmonize_features

class MLService:
    """Service pour gérer les modèles ML."""
    
    def __init__(self):
        """Initialise le service ML avec les vrais modèles."""
        # On n'instancie plus PriceModel ici, on utilise le modèle du cache
        # Charger tous les modèles SARIMAX (un par cluster)
        self.sarimax_models: Dict[int, TrendModel] = {}
        sarimax_pattern = "api_test/models/best_sarimax_cluster*_parallel.joblib"
        for path in glob.glob(sarimax_pattern):
            # Extraire le numéro de cluster du nom de fichier
            cluster_id = int(path.split("cluster")[-1].split("_")[0])
            self.sarimax_models[cluster_id] = TrendModel(path)
    
    def predict_price(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fait une prédiction de prix avec le vrai modèle.
        """
        model = get_model()
        preprocessor = get_preprocessor()
        # Récupère la liste des colonnes d'entrée attendues par le pipeline
        pipeline_input_cols = list(preprocessor.feature_names_in_)
        # Filtre strictement les colonnes d'entrée
        input_data = {k: v for k, v in input_data.items() if k in pipeline_input_cols}
        # Complète les colonnes manquantes avec une valeur par défaut
        for col in pipeline_input_cols:
            if col not in input_data:
                input_data[col] = ""
        # Conversion explicite en DataFrame
        X = pd.DataFrame([input_data])
        # Correction : conversion systématique en float (tout ce qui n'est pas convertible devient NaN)
        X = X.apply(pd.to_numeric, errors='coerce')
        X_trans = preprocessor.transform(X)
        # Correction avancée : trouver la bonne étape pour get_feature_names_out
        def get_last_feature_names_out(estimator, X):
            # Si c'est un pipeline, on cherche la dernière étape qui a get_feature_names_out
            if hasattr(estimator, 'steps'):
                for name, step in reversed(estimator.steps):
                    if hasattr(step, 'get_feature_names_out'):
                        try:
                            return step.get_feature_names_out()
                        except Exception:
                            continue
            if hasattr(estimator, 'get_feature_names_out'):
                try:
                    return estimator.get_feature_names_out()
                except Exception:
                    pass
            # fallback : colonnes numériques
            return np.arange(X.shape[1])

        feature_names = get_last_feature_names_out(preprocessor, X)
        if len(feature_names) == X_trans.shape[1]:
            X_trans = pd.DataFrame(X_trans, columns=feature_names)
        else:
            X_trans = pd.DataFrame(X_trans)
        model_features = get_model_metadata()["feature_names"]
        # Harmonisation des features après le pipeline
        X_harmonized = harmonize_features(X_trans, model_features)
        surface = input_data.get('surface', 1)
        try:
            surface = float(surface)
            if surface <= 0:
                surface = 1
        except Exception:
            surface = 1
        if hasattr(model, 'prepare_features'):
            y_pred = model.predict(X_harmonized.iloc[0].to_dict())
            print('DEBUG y_pred:', y_pred)
            # Si la sortie est un dict (cas PriceModel), on extrait la valeur float
            if isinstance(y_pred, dict):
                prix_m2 = float(y_pred.get("estimation", 0))
                intervalle = y_pred.get("intervalle_confiance", {"min": prix_m2*0.95, "max": prix_m2*1.05})
            else:
                prix_m2 = float(y_pred[0])
                intervalle = {"min": prix_m2*0.95, "max": prix_m2*1.05}
            prix_total = prix_m2 * surface
            return {
                "estimation": prix_total,
                "prix_m2": prix_m2,
                "intervalle_confiance": {
                    "min": float(intervalle["min"]) * surface,
                    "max": float(intervalle["max"]) * surface
                }
            }
        else:
            y_pred = model.predict(X_harmonized)
            print('DEBUG y_pred:', y_pred)
            prix_m2 = float(y_pred[0])
            prix_total = prix_m2 * surface
            return {
                "estimation": prix_total,
                "prix_m2": prix_m2,
                "intervalle_confiance": {"min": prix_m2 * 0.95 * surface, "max": prix_m2 * 1.05 * surface}
            }
    
    def predict_trend(self, input_data: Dict[str, Any], cluster_id: int = 0) -> Dict[str, Any]:
        """
        Prédit la tendance des prix avec le modèle SARIMAX du cluster donné.
        """
        if cluster_id not in self.sarimax_models:
            raise ValueError(f"Aucun modèle SARIMAX pour le cluster {cluster_id}")
        return self.sarimax_models[cluster_id].predict_trend(input_data)

# Instance globale du service
ml_service = MLService() 