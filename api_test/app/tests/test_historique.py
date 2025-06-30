import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.main import app

client = TestClient(app)

# À remplacer par une clé API valide pour les tests
API_KEY = "test_api_key"

def test_get_estimations_unauthorized():
    """Test sans clé API : doit retourner 422 ou 401 selon la logique"""
    response = client.get("/api/v1/estimations")
    assert response.status_code in (401, 422)

def test_get_estimations_authorized(monkeypatch):
    """Test avec clé API valide et mocks pour la base de données"""

    def fake_get_estimations(db, limit, offset):
        class FakeObj:
            id_estimation = 1
            date_estimation = __import__("datetime").datetime.now()
            bien = {"type": "appartement", "surface": 50}
            localisation = {"code_postal": "75000"}
            estimation = {"prix": 300000, "indice_confiance": 0.8}
        return [FakeObj()]

    def fake_count_estimations(db):
        return 1

    # Patch les fonctions utilisées dans la route
    import api_test.api.historique as historique_module
    monkeypatch.setattr(historique_module, "get_estimations", fake_get_estimations)
    monkeypatch.setattr(historique_module, "count_estimations", fake_count_estimations)
    monkeypatch.setattr(historique_module, "verify_api_key", lambda x: True)

    headers = {"X-API-Key": API_KEY}
    response = client.get("/api/v1/estimations", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "estimations" in data
    assert "estimation_metadata" in data
    assert isinstance(data["estimations"], list)
    assert data["estimation_metadata"]["total"] == 1