"""Main module for real estate price modeling using time series."""

import argparse
import logging
import os
import sys

import yaml

from src.compagnon_immo.evaluation.evaluator import ModelEvaluator
from src.compagnon_immo.training.trainer import ImmoTrainer

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    """Configuration du logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str = 'config/model_config.yaml'):
    """Charger la configuration."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description='Modélisation des prix immobiliers par séries temporelles'
    )
    parser.add_argument(
        '--data', required=True, help='Chemin vers le fichier de données'
    )
    parser.add_argument(
        '--config', default='config/config.yaml',
        help='Chemin vers le fichier de configuration'
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Vérifier que les fichiers existent
    if not os.path.exists(args.data):
        logger.error("Fichier de données non trouvé: %s", args.data)
        return

    if not os.path.exists(args.config):
        logger.error("Fichier de configuration non trouvé: %s", args.config)
        return

    # Créer les dossiers de sortie
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    try:
        # Entraîner le modèle
        trainer = ImmoTrainer(args.config)
        results = trainer.run_training(args.data)

        # Afficher les résultats
        logger.info("=== RÉSULTATS DE L'ENTRAÎNEMENT ===")
        for metric, value in results['metrics'].items():
            logger.info("%s: %.4f", metric.upper(), value)

        # Générer les visualisations
        evaluator = ModelEvaluator()
        evaluator.plot_predictions(
            results['actual'],
            results['predictions'],
            'outputs/plots/predictions.png'
        )
        evaluator.plot_training_history(
            results['history'],
            'outputs/plots/training_history.png'
        )

        logger.info(
            "Entraînement terminé. Résultats sauvegardés dans 'outputs/'"
        )

    except Exception as e:
        logger.error("Erreur: %s", e)
        raise


if __name__ == "__main__":
    main()
