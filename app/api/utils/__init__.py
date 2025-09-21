# app/api/utils/__init__.py
"""
Package utils : exporte le MODULE 'model_loader' (avec .get_model, .get_preprocessor, etc.)
+ ré-exporte aussi les fonctions au besoin.
"""

# 1) importer le module pour que 'from app.api.utils import model_loader' donne bien un module
from . import model_loader as model_loader  # noqa: F401

# 2) ré-exporter les fonctions pour 'from app.api.utils import get_model'
from .model_loader import (  # noqa: F401
    model_loader as load_model_dict,  # alias optionnel si besoin
    get_model,
    get_preprocessor,
    get_model_metadata,
)

__all__ = [
    "model_loader",           # le module
    "load_model_dict",        # la fonction dict optionnelle
    "get_model",
    "get_preprocessor",
    "get_model_metadata",
]
