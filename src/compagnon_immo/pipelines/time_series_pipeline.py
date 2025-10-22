import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from ..models.model_factory import ModelFactory
from ..data.preprocessor import DataPreprocessor
from ..evaluation.metrics import calculate_time_series_metrics
import logging

class TimeSeriesPipeline:
    """Pipeline principale pour la modélisation de séries temporelles."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Exécuter la pipeline complète."""
        try:
            # 1. Préparation des données
            self.logger.info("Préparation des données pour séries temporelles...")
            processed_data = self._prepare_time_series_data(data)
            
            # 2. Division des données
            train_data, test_data = self._split_data(processed_data)
            
            # 3. Création et entraînement du modèle
            self.logger.info(f"Création du modèle {self.config['model_type']}...")
            self.model = ModelFactory.create_model(
                self.config['model_type'],
                **self.config.get('model_params', {})
            )
            
            # 4. Préparation des séquences
            X_train, y_train = self.model.prepare_sequences(train_data)
            X_test, y_test = self.model.prepare_sequences(test_data)
            
            # 5. Entraînement
            self.logger.info("Entraînement du modèle...")
            history = self.model.fit(
                X_train, y_train,
                **self.config.get('training_params', {})
            )
            
            # 6. Prédictions et évaluation
            predictions = self.model.predict(X_test)
            metrics = calculate_time_series_metrics(y_test, predictions)
            
            return {
                'model': self.model,
                'predictions': predictions,
                'actual': y_test,
                'metrics': metrics,
                'history': history.history if history else None
            }
            
        except Exception as e:
            self.logger.error(f"Erreur dans la pipeline: {str(e)}")
            raise
    
    def _prepare_time_series_data(self, data: pd.DataFrame) -> np.ndarray:
        """Préparer les données pour les séries temporelles."""
        # Sélectionner la colonne cible (prix)
        target_column = self.config.get('target_column', 'prix')
        if target_column not in data.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée dans les données.")
        
        # Trier par date si disponible
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        return data[target_column].values.reshape(-1, 1)
    
    def _split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Diviser les données en ensemble d'entraînement et de test."""
        split_ratio = self.config.get('train_split', 0.8)
        split_index = int(len(data) * split_ratio)
        return data[:split_index], data[split_index:]
