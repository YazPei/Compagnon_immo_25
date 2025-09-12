"""
Module pour le géocodage.
"""

import requests
from typing import Optional, Dict


def geocode_address(code_postal: str, ville: str, quartier: Optional[str] = None) -> Dict[str, float]:
    """
    Géocode une adresse en latitude et longitude.

    Args:
        code_postal (str): Code postal de l'adresse.
        ville (str): Ville de l'adresse.
        quartier (Optional[str]): Quartier ou complément d'adresse (facultatif).

    Returns:
        dict: Dictionnaire contenant la latitude et la longitude.
    """
    # Mock : retourne des coordonnées fixes pour les tests
    # Remplacez cette partie par une requête à une API de géocodage si nécessaire
    return {
        "latitude": 48.8566,
        "longitude": 2.3522
    }


def reverse_geocode(latitude: float, longitude: float) -> str:
    """
    Récupère le code postal à partir des coordonnées.

    Args:
        latitude (float): Latitude des coordonnées.
        longitude (float): Longitude des coordonnées.

    Returns:
        str: Code postal correspondant aux coordonnées.
    """
    # Mock pour les tests
    # Remplacez cette partie par une requête à une API de géocodage inversé si nécessaire
    return "75001"