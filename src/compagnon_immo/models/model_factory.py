from .time_series_models import LSTMModel, GRUModel
from typing import Dict, Any

class ModelFactory:
    """Factory pour créer des modèles de séries temporelles uniquement."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        """Créer un modèle de série temporelle."""
        models = {
            'lstm': LSTMModel,
            'gru': GRUModel
        }
        
        if model_type.lower() not in models:
            raise ValueError(f"Type de modèle non supporté: {model_type}. Types disponibles: {list(models.keys())}")
        
        return models[model_type.lower()](**kwargs)
    
    @staticmethod
    def get_available_models():
        """Retourner la liste des modèles disponibles."""
        return ['lstm', 'gru']
