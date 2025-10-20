from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)
API_KEY = "test_api_key"

EXAMPLES: List[Dict[str, Any]] = [
    {
        "bien": {
            "type": "appartement",
            "surface": 65,
            "nb_pieces": 3,
            "nb_chambres": 2,
            "etage": 2,
            "annee_construction": 1985,
            "etat_general": "bon",
            "exposition": "sud",
            "ascenseur": True,
            "balcon": True,
            "terrasse": False,
            "parking": False,
            "cave": True,
            "dpe": "D",
        },
        "localisation": {
            "code_postal": "75015",
            "ville": "Paris",
            "quartier": "Vaugirard",
        },
        "transaction": {"type": "vente"},
    },
    {
        "bien": {
            "type": "maison",
            "surface": 120,
            "nb_pieces": 5,
            "nb_chambres": 4,
            "etage": 0,
            "annee_construction": 2002,
            "etat_general": "neuf",
            "exposition": "est",
            "ascenseur": False,
            "balcon": False,
            "terrasse": True,
            "surface_exterieure": 40,
            "parking": True,
            "cave": False,
            "dpe": "B",
        },
        "localisation": {
            "code_postal": "69008",
            "ville": "Lyon",
            "quartier": "Monplaisir",
        },
        "transaction": {"type": "vente"},
    },
    {
        "bien": {
            "type": "studio",
            "surface": 22,
            "nb_pieces": 1,
            "nb_chambres": 0,
            "etage": 4,
            "annee_construction": 1970,
            "etat_general": "moyen",
            "exposition": "ouest",
            "ascenseur": False,
            "balcon": False,
            "terrasse": False,
            "parking": False,
            "cave": False,
        },
        "localisation": {"code_postal": "13006", "ville": "Marseille"},
        "transaction": {"type": "location"},
    },
    {
        "bien": {
            "type": "loft",
            "surface": 90,
            "nb_pieces": 2,
            "nb_chambres": 1,
            "etage": 1,
            "annee_construction": 2010,
            "etat_general": "bon",
            "exposition": "nord-est",
            "ascenseur": True,
            "balcon": False,
            "terrasse": True,
            "surface_exterieure": 20,
            "parking": True,
            "cave": False,
            "dpe": "C",
        },
        "localisation": {
            "code_postal": "33000",
            "ville": "Bordeaux",
            "latitude": 44.8378,
            "longitude": -0.5792,
        },
        "transaction": {"type": "vente"},
    },
]


@pytest.mark.parametrize("payload", EXAMPLES)
def test_estimation(payload: Dict[str, Any]):
    """Test de l'endpoint d'estimation avec différents payloads."""
    response = client.post(
        "/api/v1/estimation", json=payload, headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200

    data = response.json()

    # Vérifier la structure attendue
    assert "estimation" in data
    assert "prix" in data["estimation"]
    assert data["estimation"]["prix"] >= 0
    assert "input" in data

    # Vérifier que indice_confiance est bien un float
    assert isinstance(data["estimation"]["indice_confiance"], float)
    assert 0 <= data["estimation"]["indice_confiance"] <= 1


def test_estimation_unauthorized():
    """Test de l'endpoint d'estimation sans authentification."""
    response = client.post(
        "/api/v1/estimation",
        json={"surface": 100, "nb_pieces": 4, "code_postal": "75001"},
    )
    assert response.status_code == 401
