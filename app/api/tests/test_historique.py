from datetime import datetime
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)
API_KEY = "test_api_key"


def test_get_estimations_unauthorized():
    """Sans clé API -> 422/401 selon dépendance."""
    response = client.get("/api/v1/historique/estimations")
    assert response.status_code in {401, 422}


def test_get_estimations_authorized():
    """Endpoint avec clé API valide et mocks DB."""

    def fake_get_estimations(db: Any, limit: int, offset: int):
        class FakeObj:
            id_estimation = "test-123"
            date_estimation = datetime.now()
            bien: dict[str, Any] = {"type": "appartement", "surface": 50}
            localisation: dict[str, Any] = {"code_postal": "75000"}
            estimation: dict[str, Any] = {
                "prix": 300000,
                "indice_confiance": 0.8,
            }

        return [FakeObj()]

    def fake_count_estimations(db: Any) -> int:
        return 1

    with patch(
        "app.api.routes.historique.get_estimations",
        side_effect=fake_get_estimations,
    ):
        with patch(
            "app.api.routes.historique.count_estimations",
            side_effect=fake_count_estimations,
        ):
            response = client.get(
                "/api/v1/historique/estimations", headers={"X-API-Key": API_KEY}
            )
            assert response.status_code == 200
            data = response.json()
            assert "estimations" in data
            assert isinstance(data["estimations"], list)
            assert len(data["estimations"]) == 1
            assert "total" in data
            assert data["total"] == 1
            assert "page" in data
            assert "limite" in data
