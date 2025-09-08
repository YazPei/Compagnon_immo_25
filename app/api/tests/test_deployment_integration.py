"""Tests d'intégration pour le déploiement."""

import pytest
import requests
import os
import subprocess
import time
from unittest.mock import patch, MagicMock


class TestDeploymentIntegration:
    """Tests d'intégration pour vérifier le déploiement et les performances."""

    @pytest.fixture
    def api_base_url(self):
        """URL de base pour les tests."""
        return os.getenv("API_BASE_URL", "http://localhost:8000")

    @pytest.fixture
    def api_headers(self):
        """Headers pour les requêtes API."""
        return {
            "Content-Type": "application/json",
            "X-API-Key": "test-key-123"
        }

    def test_deployment_health_endpoint(self, api_base_url):
        """Test de l'endpoint de santé du déploiement."""
        response = requests.get(f"{api_base_url}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_deployment_status_endpoint(self, api_base_url, api_headers):
        """Test de l'endpoint de statut du déploiement."""
        response = requests.get(
            f"{api_base_url}/api/v1/deployment/status",
            headers=api_headers
        )
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy", "error"]

    @patch('subprocess.run')
    def test_deployment_script_execution(self, mock_subprocess):
        """Test d'exécution du script de déploiement."""
        # Mock du résultat de subprocess
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Deployment successful",
            stderr=""
        )

        # Import et test de la fonction de déploiement
        from ...scripts.test_deployment import DeploymentTester

        tester = DeploymentTester("http://localhost:8000")

        # Test de santé
        with patch('requests.get') as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "ok"}
            )

            result = tester.test_api_health()
            assert result is True

    def test_deployment_script_imports(self):
        """Test que le script de déploiement peut être importé."""
        try:
            # Import des modules du script de test de déploiement
            import sys
            sys.path.append('/app/scripts')

            from test_deployment import DeploymentTester

            # Vérifier que la classe peut être instanciée
            tester = DeploymentTester("http://localhost:8000")
            assert tester.api_url == "http://localhost:8000"
            assert tester.test_results == []

        except ImportError as e:
            pytest.fail(f"Impossible d'importer le script de test: {e}")

    @pytest.mark.slow
    def test_full_api_workflow(self, api_base_url, api_headers):
        """Test complet du workflow API."""
        # 1. Test de santé
        health_response = requests.get(f"{api_base_url}/health")
        assert health_response.status_code == 200

        # 2. Test d'estimation
        estimation_payload = {
            "bien": {
                "type": "appartement",
                "surface": 70,
                "nb_pieces": 3,
                "nb_chambres": 2
            },
            "localisation": {
                "code_postal": "75001",
                "ville": "Paris"
            },
            "transaction": {
                "type": "vente"
            }
        }

        estimation_response = requests.post(
            f"{api_base_url}/api/v1/estimation",
            json=estimation_payload,
            headers=api_headers
        )

        assert estimation_response.status_code == 200
        data = estimation_response.json()
        assert "estimation" in data
        assert "prix" in data["estimation"]

        # 3. Test de documentation
        docs_response = requests.get(f"{api_base_url}/docs")
        assert docs_response.status_code == 200

    def test_performance_requirements(self, api_base_url):
        """Test des exigences de performance."""
        # Test de temps de réponse
        start_time = time.time()
        response = requests.get(f"{api_base_url}/health")
        response_time = time.time() - start_time

        assert response.status_code == 200
        assert response_time < 5.0, f"Temps de réponse trop lent: {response_time}s"

        # Test de charge légère
        response_times = []
        for _ in range(10):
            start = time.time()
            resp = requests.get(f"{api_base_url}/health")
            response_times.append(time.time() - start)
            assert resp.status_code == 200

        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 2.0, f"Temps de réponse moyen trop lent: {avg_response_time}s"


# Commande pour exécuter ces tests spécifiquement
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])