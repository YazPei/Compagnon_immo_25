"""
Module pour l'enrichissement des features.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError


# Modèle Pydantic pour valider les entrées utilisateur
class FeatureInput(BaseModel):
    code_postal: str = Field(
        ..., pattern=r"^\d{5}$", description="Code postal à 5 chiffres."
    )
    ville: str
    quartier: Optional[str] = None


def validate_input(data: Dict[str, Any]) -> FeatureInput:
    """
    Valide les entrées utilisateur.

    Args:
        data (dict): Données à valider.

    Returns:
        FeatureInput: Données validées.

    Raises:
        ValidationError: Si les données ne sont pas valides.
    """
    try:
        return FeatureInput(**data)
    except ValidationError as e:
        raise ValueError(f"Entrées invalides : {e}")
