"""
Script pour nettoyer les artefacts inutiles générés par les pipelines.
"""

import os
import shutil
from pathlib import Path


def cleanup_directory(directory: str, extensions: list):
    """
    Supprime les fichiers avec des extensions spécifiques dans un répertoire.

    Args:
        directory (str): Chemin du répertoire.
        extensions (list): Liste des extensions de fichiers à supprimer.
    """
    path = Path(directory)
    if not path.exists():
        print(f"Répertoire introuvable : {directory}")
        return

    for ext in extensions:
        for file in path.glob(f"*{ext}"):
            try:
                file.unlink()
                print(f"Supprimé : {file}")
            except Exception as e:
                print(f"Erreur lors de la suppression de {file} : {e}")


def main():
    # Répertoires à nettoyer
    directories = [
        "data/processed",
        "logs",
        "mlruns"
    ]

    # Extensions à supprimer
    extensions = [".tmp", ".log", ".bak"]

    for directory in directories:
        cleanup_directory(directory, extensions)


if __name__ == "__main__":
    main()