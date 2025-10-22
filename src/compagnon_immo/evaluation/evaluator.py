"""
Module d'évaluation des modèles pour Compagnon Immo.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import os
import logging


class ModelEvaluator:
    """Classe pour évaluer les performances des modèles."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcule les métriques d'évaluation."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        self.logger.info(f"Métriques calculées: {metrics}")
        return metrics

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """Trace les prédictions vs valeurs réelles."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title('Prédictions vs Valeurs Réelles')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Graphique sauvegardé: {save_path}")

        plt.show()

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """Trace les résidus."""
        residuals = y_true - y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Prédictions')
        plt.ylabel('Résidus')
        plt.title('Analyse des Résidus')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Graphique des résidus sauvegardé: {save_path}")

        plt.show()

    def plot_training_history(self, history: Dict, save_path: str = None):
        """Trace l'historique d'entraînement."""
        if not history:
            self.logger.warning("Aucun historique d'entraînement fourni")
            return

        plt.figure(figsize=(12, 4))

        # Loss
        if 'loss' in history:
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Évolution de la Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        # Metrics
        if 'mae' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['mae'], label='Train MAE')
            if 'val_mae' in history:
                plt.plot(history['val_mae'], label='Validation MAE')
            plt.title('Évolution de la MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Historique d'entraînement sauvegardé: {save_path}")

        plt.show()

    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Modèle") -> str:
        """Génère un rapport d'évaluation."""
        metrics = self.calculate_metrics(y_true, y_pred)

        report = f"""
Rapport d'Évaluation - {model_name}
{'='*50}

Métriques de Performance:
- MSE (Mean Squared Error): {metrics['MSE']:.4f}
- RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}
- MAE (Mean Absolute Error): {metrics['MAE']:.4f}
- R² Score: {metrics['R2']:.4f}

Interprétation R²:
- R² = 1.0: Prédictions parfaites
- R² = 0.0: Prédictions équivalentes à la moyenne
- R² < 0.0: Prédictions pires que la moyenne

Statistiques des Données:
- Nombre d'échantillons: {len(y_true)}
- Plage des valeurs réelles: [{y_true.min():.2f}, {y_true.max():.2f}]
- Moyenne des valeurs réelles: {y_true.mean():.2f}
- Écart-type des valeurs réelles: {y_true.std():.2f}
"""

        self.logger.info(f"Rapport généré pour {model_name}")
        return report
