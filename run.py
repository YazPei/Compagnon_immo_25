#!/usr/bin/env python3
"""
Script de lancement robuste pour la modélisation des prix immobiliers.
Usage: python run.py --data data/prices.csv
"""

import os
import sys
import logging
from pathlib import Path

# Ajouter le répertoire du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def check_dependencies():
    """Vérifier que toutes les dépendances sont installées."""
    required_packages = [
        'tensorflow', 'sklearn', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'pyyaml', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Packages manquants: {', '.join(missing_packages)}")
        print("Installez-les avec: pip install -r requirements.txt")
        sys.exit(1)

def setup_directories():
    """Créer les dossiers nécessaires."""
    directories = [
        'outputs',
        'outputs/plots',
        'outputs/models',
        'data',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Vérifications préliminaires
    check_dependencies()
    setup_directories()
    
    # Lancer le programme principal
    from main import main
    main()
