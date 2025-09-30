# app/api/utils/__init__.py
"""
Package utils : exporte le MODULE 'model_loader' (avec .get_model, .get_preprocessor, etc.)
+ ré-exporte aussi les fonctions au besoin.
"""

# 1) importer le module pour que 'from app.api.utils import model_loader' donne bien un module
from . import model_loader as model_loader  # noqa: F401
# 2) ré-exporter les fonctions pour 'from app.api.utils import get_model'
from .model_loader import get_model, get_model_metadata, get_preprocessor
from .model_loader import \
    model_loader as load_model_dict  # noqa: F401; alias optionnel si besoin

__all__ = [
    "model_loader",  # le module
    "load_model_dict",  # la fonction dict optionnelle
    "get_model",
    "get_preprocessor",
    "get_model_metadata",
]
