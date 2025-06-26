"""
Module contenant la classe TrendModel pour la prédiction des tendances de prix immobiliers.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

class TrendModel:
    """Classe pour gérer le modèle de prédiction des tendances de prix."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise le modèle de prédiction des tendances.
        
        Args:
            model_path: Chemin vers le fichier .joblib du modèle SARIMAX
        """
        self.model: Optional[SARIMAXResults] = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Charge le modèle depuis un fichier .joblib
        
        Args:
            model_path: Chemin vers le fichier .joblib
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le modèle n'est pas valide
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier modèle {model_path} n'existe pas")
        
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du modèle: {str(e)}")
    
    def prepare_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prépare les données pour la prédiction.
        
        Args:
            input_data: Dictionnaire contenant les paramètres (localisation, période, etc.)
            
        Returns:
            DataFrame avec les données préparées
        """
        # TODO: Implémenter la préparation des données selon votre logique métier
        # Exemple: Créer un index temporel pour la prédiction
        start_date = pd.to_datetime(input_data.get('start_date', pd.Timestamp.now()))
        periods = input_data.get('periods', 12)  # Par défaut 12 mois
        
        date_index = pd.date_range(start=start_date, periods=periods, freq='ME')
        return pd.DataFrame(index=date_index)
    
    def predict_trend(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prédit la tendance des prix.
        
        Args:
            input_data: Dictionnaire contenant les paramètres de prédiction
            
        Returns:
            Dict contenant les prédictions et intervalles de confiance
            
        Raises:
            ValueError: Si le modèle n'est pas chargé ou si les données sont invalides
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé")
        
        try:
            # Préparer les données
            data = self.prepare_data(input_data)
            # Récupérer exog si fourni
            exog = input_data.get('exog', None)
            # Faire la prédiction avec intervalle de confiance
            if exog is not None:
                forecast = self.model.get_forecast(steps=len(data), exog=exog)
            else:
                forecast = self.model.get_forecast(steps=len(data))
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            # Formater les résultats
            results = []
            
            # Si conf_int est un numpy.ndarray, le convertir en DataFrame
            if isinstance(conf_int, np.ndarray):
                conf_int = pd.DataFrame(
                    conf_int,
                    columns=['lower', 'upper'],
                    index=data.index
                )

            # Recherche robuste des colonnes 'lower' et 'upper'
            if isinstance(conf_int, pd.DataFrame):
                lower_col = next((col for col in conf_int.columns if str(col).lower().startswith('lower')), conf_int.columns[0])
                upper_col = next((col for col in conf_int.columns if str(col).lower().startswith('upper')), conf_int.columns[1])
                lower_vals = conf_int[lower_col]
                upper_vals = conf_int[upper_col]
            else:
                lower_vals = conf_int[:, 0]
                upper_vals = conf_int[:, 1]

            for date, pred, lower, upper in zip(
                data.index,
                mean_forecast,
                lower_vals,
                upper_vals
            ):
                results.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "prediction": float(pred),
                    "intervalle_confiance": {
                        "min": float(lower),
                        "max": float(upper)
                    }
                })
            
            # Calcul de l'évolution annuelle en pourcentage
            if len(results) > 1 and results[0]["prediction"] != 0:
                evolution_annuelle = 100 * (results[-1]["prediction"] - results[0]["prediction"]) / results[0]["prediction"]
            else:
                evolution_annuelle = 0
            
            return {
                "tendance": results,
                "metadata": {
                    "start_date": data.index[0].strftime("%Y-%m-%d"),
                    "end_date": data.index[-1].strftime("%Y-%m-%d"),
                    "nombre_periodes": len(data),
                    "evolution_annuelle": evolution_annuelle
                }
            }
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction: {str(e)}") 