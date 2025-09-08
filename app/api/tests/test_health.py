import pytest
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

    def test_health_check_detailed(self):
        """Test de l'endpoint de health check détaillé."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "critical"]

        # Vérifier les composants dans le health check détaillé
        assert "components" in data
        components = data["components"]
        assert isinstance(components, dict)

        # Vérifier des composants spécifiques
        assert "api" in components
        assert "database" in components
        assert "ml_service" in components
        assert components["api"]["status"] in ["healthy", "degraded", "critical"]
        assert components["database"]["status"] in ["healthy", "critical"]
        assert components["ml_service"]["status"] in ["healthy", "degraded", "critical"]

    def test_health_check_response_time(self):
        """Test du temps de réponse de l'endpoint de health check."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        # Vérifier que le temps de réponse est raisonnable
        assert response.elapsed.total_seconds() < 1.0, (
            f"Temps de réponse trop long : {response.elapsed.total_seconds()}s"
        )
