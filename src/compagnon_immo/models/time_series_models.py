import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import joblib

class BaseTimeSeriesModel:
    """Classe de base pour les modèles de séries temporelles."""
    
    def __init__(self, sequence_length: int = 60, **kwargs):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Préparer les séquences pour l'entraînement."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32):
        """Entraîner le modèle."""
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Compilation du modèle
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Entraînement
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions.")
        
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions_scaled = self.model.predict(X_scaled)
        return self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

class LSTMModel(BaseTimeSeriesModel):
    """Modèle LSTM pour les séries temporelles."""
    
    def __init__(self, sequence_length: int = 60, units: int = 50, dropout: float = 0.2, **kwargs):
        super().__init__(sequence_length, **kwargs)
        self.units = units
        self.dropout = dropout
        self._build_model()
    
    def _build_model(self):
        """Construire le modèle LSTM."""
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(self.dropout),
            LSTM(self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(25),
            Dense(1)
        ])

class GRUModel(BaseTimeSeriesModel):
    """Modèle GRU pour les séries temporelles."""
    
    def __init__(self, sequence_length: int = 60, units: int = 50, dropout: float = 0.2, **kwargs):
        super().__init__(sequence_length, **kwargs)
        self.units = units
        self.dropout = dropout
        self._build_model()
    
    def _build_model(self):
        """Construire le modèle GRU."""
        self.model = Sequential([
            GRU(self.units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(self.dropout),
            GRU(self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(25),
            Dense(1)
        ])
