"""
Configuration centralisée des logs pour le projet.
"""

import logging
import logging.config
import os
import sys
from typing import Any, Dict

from loguru import logger

# Définir les niveaux de log par environnement
ENV = os.getenv("ENV", "development")
LOG_LEVEL = "DEBUG" if ENV == "development" else "INFO"

# Configuration des logs
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            ),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": LOG_LEVEL,
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "level": "DEBUG",
            "filename": "logs/app.log",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": LOG_LEVEL,
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "app": {
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
}

# Appliquer la configuration des logs
logging.config.dictConfig(LOGGING_CONFIG)

# Supprimer les gestionnaires par défaut de Loguru pour éviter les doublons
logger.remove()

# Ajouter un gestionnaire pour les logs structurés
logger.add(
    sys.stdout,
    format="{{'time': '{time}', 'level': '{level}', 'message': '{message}'}}",
    level=LOG_LEVEL,
    serialize=True,
)

# Exemple d'utilisation :
logger.info("Loguru est configuré pour des logs structurés.")
