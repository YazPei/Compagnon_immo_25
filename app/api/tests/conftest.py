import pytest
from fastapi.testclient import TestClient
import asyncio
from typing import Any, Dict

from app.api.main import app
from app.api.security.auth import auth_manager


@pytest.fixture
def test_client():
    """Fixture pour créer un client de test."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Fixture pour générer des headers d'authentification pour les tests."""
    try:
        token = auth_manager.create_access_token(data={"sub": "testuser"})
    except Exception:
        token = "test_token_for_unit_tests"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session")
def event_loop():
    """Créer un event loop pour les tests asyncio."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def client():
    """Client de test FastAPI."""
    return TestClient(app)


@pytest.fixture
def api_key():
    """Clé API valide pour les tests."""
    return "test_api_key"


@pytest.fixture
def test_estimation_payload() -> Dict[str, Any]:
    """Payload de test pour les estimations."""
    return {
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
        "transaction": {
            "type": "vente",
        },
    }


@pytest.fixture
def mock_estimation_response() -> Dict[str, Any]:
    """Réponse simulée pour une estimation."""
    return {
        "estimation": {
            "prix": 450000,
            "prix_min": 405000,
            "prix_max": 495000,
            "prix_m2": 6923.08,
            "indice_confiance": 0.85,
        },
        "marche": {
            "prix_moyen_quartier": 440000,
            "evolution_annuelle": 2.5,
            "delai_vente_moyen": 45,
        },
        "metadata": {
            "id_estimation": "est-20250907-12345678",
            "date_estimation": "2025-09-07T12:34:56",
            "version_modele": "1.0.0",
        },
    }
