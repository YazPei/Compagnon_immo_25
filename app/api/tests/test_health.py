from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Tests pour les endpoints de health check."""

    def test_health_check(self):
        """Test de l'endpoint de health check basique."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ok", "healthy"]

    @patch("app.api.services.health_service.check_database")
    @patch("app.api.services.health_service.check_ml_service")
    def test_health_check_detailed(self, mock_check_ml_service, mock_check_database):
        """Test de l'endpoint de health check détaillé."""
        mock_check_database.return_value = {"status": "healthy"}
        mock_check_ml_service.return_value = {"status": "healthy"}

        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "critical"]
        assert "components" in data
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["ml_service"]["status"] == "healthy"
