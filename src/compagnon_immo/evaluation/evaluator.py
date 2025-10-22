"""Module d'évaluation des modèles pour Compagnon Immo."""

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Classe pour évaluer les performances des modèles."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict[str, float]:
        """Calcule les métriques d'évaluation."""
        mse: float = mean_squared_error(y_true, y_pred)
        rmse: float = np.sqrt(mse)  # type: ignore
        mae: float = mean_absolute_error(y_true, y_pred)
        r2: float = r2_score(y_true, y_pred)

        metrics: Dict[str, float] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        self.logger.info("Métriques calculées: %s", metrics)
        return metrics

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: str | None = None):
        """Trace les prédictions vs valeurs réelles."""
        plt.figure(figsize=(10, 6))  # type: ignore
        plt.scatter(y_true, y_pred, alpha=0.5)  # type: ignore
        plt.plot(  # type: ignore
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'r--', lw=2
        )  # type: ignore
        plt.xlabel('Valeurs Réelles')  # type: ignore
        plt.ylabel('Prédictions')  # type: ignore
        plt.title('Prédictions vs Valeurs Réelles')  # type: ignore
        plt.grid(True)  # type: ignore

        if save_path:
            plt.savefig(  # type: ignore
                save_path,
                dpi=300,
                bbox_inches='tight'
            )  # type: ignore[reportUnknownMemberType]
            self.logger.info(
                "Graphique sauvegardé: %s",
                save_path
            )

        plt.show()  # type: ignore

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                       save_path: str | None = None):
        """Trace les résidus."""
        residuals: np.ndarray = y_true - y_pred

        plt.figure(figsize=(10, 6))  # type: ignore
        plt.scatter(y_pred, residuals, alpha=0.5)  # type: ignore
        plt.axhline(y=0, color='r', linestyle='--')  # type: ignore
        plt.xlabel('Prédictions')  # type: ignore
        plt.ylabel('Résidus')  # type: ignore
        plt.title('Analyse des Résidus')  # type: ignore
        plt.grid(True)  # type: ignore

        if save_path:
            plt.savefig(  # type: ignore
                save_path,
                dpi=300,
                bbox_inches='tight'
            )  # type: ignore[reportUnknownMemberType]
            self.logger.info(
                "Graphique des résidus sauvegardé: %s",
                save_path
            )

        plt.show()  # type: ignore

    def plot_training_history(self, history: Dict[str, List[float]],
                              save_path: str | None = None):
        """Trace l'historique d'entraînement."""
        if not history:
            self.logger.warning("Aucun historique d'entraînement fourni")
            return

        plt.figure(figsize=(12, 4))  # type: ignore

        # Loss
        if 'loss' in history:
            plt.subplot(1, 2, 1)  # type: ignore
            plt.plot(history['loss'], label='Train Loss')  # type: ignore
            if 'val_loss' in history:
                plt.plot(  # type: ignore
                    history['val_loss'],
                    label='Validation Loss'
                )  # type: ignore[reportUnknownMemberType]
            plt.title('Évolution de la Loss')  # type: ignore
            plt.xlabel('Epoch')  # type: ignore
            plt.ylabel('Loss')  # type: ignore
            plt.legend()  # type: ignore
            plt.grid(True)  # type: ignore

        # Metrics
        if 'mae' in history:
            plt.subplot(1, 2, 2)  # type: ignore
            plt.plot(history['mae'], label='Train MAE')  # type: ignore
            if 'val_mae' in history:
                plt.plot(  # type: ignore
                    history['val_mae'],
                    label='Validation MAE'
                )  # type: ignore[reportUnknownMemberType]
            plt.title('Évolution de la MAE')  # type: ignore
            plt.xlabel('Epoch')  # type: ignore
            plt.ylabel('MAE')  # type: ignore
            plt.legend()  # type: ignore
            plt.grid(True)  # type: ignore

        plt.tight_layout()  # type: ignore

        if save_path:
            plt.savefig(  # type: ignore
                save_path,
                dpi=300,
                bbox_inches='tight'
            )  # type: ignore[reportUnknownMemberType]
            self.logger.info(
                "Historique d'entraînement sauvegardé: %s",
                save_path
            )

        plt.show()  # type: ignore

    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str = "Modèle") -> str:
        """Génère un rapport d'évaluation."""
        metrics = self.calculate_metrics(y_true, y_pred)

        plage: str = f"[{y_true.min():.2f}, {y_true.max():.2f}]"

        std_val: float = y_true.std()

        report: str = (
            f"Rapport d'Évaluation - {model_name}\n"
            f"{'='*50}\n\n"
            "Métriques de Performance:\n"
            f"- MSE (Mean Squared Error): {metrics['MSE']:.4f}\n"
            f"- RMSE (Root Mean Squared Error): "
            f"{metrics['RMSE']:.4f}\n"
            f"- MAE (Mean Absolute Error): {metrics['MAE']:.4f}\n"
            f"- R² Score: {metrics['R2']:.4f}\n\n"
            "Interprétation R²:\n"
            "- R² = 1.0: Prédictions parfaites\n"
            "- R² = 0.0: Prédictions équivalentes à la moyenne\n"
            "- R² < 0.0: Prédictions pires que la moyenne\n\n"
            "Statistiques des Données:\n"
            f"- Nombre d'échantillons: {len(y_true)}\n"
            f"- Plage des valeurs réelles: {plage}\n"
            f"- Moyenne des valeurs réelles: {y_true.mean():.2f}\n"
            f"- Écart-type des valeurs réelles: "
            f"{std_val:.2f}\n"
        )

        self.logger.info("Rapport généré pour %s", model_name)
        return report
