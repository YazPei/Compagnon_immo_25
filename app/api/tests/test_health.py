import pytest
from fastapi.testclient import TestClient
from app.api.main import app
from unittest.mock import AsyncMock, patch
from app.api.utils.dependency_checker import check_database, check_redis, check_mlflow

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


@pytest.mark.asyncio
async def test_check_database():
    with patch("app.db.database.engine.connect") as mock_connect:
        mock_connect.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = 1
        result = await check_database()
        assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_check_redis():
    with patch("redis.from_url") as mock_redis:
        mock_redis.return_value.ping.return_value = True
        result = await check_redis("redis://localhost")
        assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_check_mlflow():
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.status_code = 200
        result = await check_mlflow("http://localhost:5000")
        assert result["status"] == "healthy"
