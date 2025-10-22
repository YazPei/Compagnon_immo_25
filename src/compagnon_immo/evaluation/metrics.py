import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculer les métriques de régression."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def calculate_time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculer les métriques pour les séries temporelles."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'r2': r2_score(y_true, y_pred)
    }

def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculer la précision directionnelle."""
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction) * 100
