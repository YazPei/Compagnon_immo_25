import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from datetime import datetime

# Ajouter le chemin racine pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import de l'application FastAPI
from app.api.main import app

client = TestClient(app)

API_KEY = "test_api_key"


def test_get_estimations_unauthorized():
    """Test de l'endpoint sans clé API - doit retourner 422 pour header manquant."""
    response = client.get("/api/v1/historique/estimations")
    # FastAPI retourne 422 quand un header requis est manquant
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"][0]["msg"] == "field required"


def test_get_estimations_authorized():
    """Test de l'endpoint avec clé API valide et mocks pour la base de données."""

    def fake_get_estimations(db, limit, offset):
        """Mock de la fonction get_estimations."""
        class FakeObj:
            id_estimation = "test-123"
            date_estimation = datetime.now()
            bien = {"type": "appartement", "surface": 50}
            localisation = {"code_postal": "75000"}
            estimation = {"prix": 300000, "indice_confiance": 0.8}
        return [FakeObj()]

    def fake_count_estimations(db):
        """Mock de la fonction count_estimations."""
        return 1

    # Patch des fonctions utilisées dans la route
    with patch('app.api.routes.historique.get_estimations', side_effect=fake_get_estimations):
        with patch('app.api.routes.historique.count_estimations', side_effect=fake_count_estimations):
            response = client.get(
                "/api/v1/historique/estimations",
                headers={"X-API-Key": API_KEY}
            )

            # Vérifications de la réponse
            assert response.status_code == 200
            data = response.json()

            # Vérifier la structure de la réponse
            assert "estimations" in data
            assert isinstance(data["estimations"], list)
            assert len(data["estimations"]) == 1

            estimation = data["estimations"][0]
            assert "id_estimation" in estimation
            assert "date_estimation" in estimation
            assert "bien" in estimation
            assert "localisation" in estimation
            assert "estimation" in estimation

            # Vérifier les autres champs
            assert "total" in data
            assert data["total"] == 1
            assert "page" in data
            assert "limite" in data