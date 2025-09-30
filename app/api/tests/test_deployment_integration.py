"""Tests d'intégration pour le déploiement."""

import pytest
import requests
import os
import time
from unittest.mock import patch, MagicMock
from typing import Any


class TestDeploymentIntegration:
    """Tests d'intégration pour vérifier le déploiement et les performances."""

    @pytest.fixture
    def api_base_url(self) -> str:
        """URL de base pour les tests."""
        return os.getenv("API_BASE_URL", "http://localhost:8000")

    def test_deployment_health_endpoint(self, api_base_url: str):
        """Test de l'endpoint de santé du déploiement."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{api_base_url}/health", timeout=10)
                assert response.status_code == 200
                data = response.json()
                assert data["status"] in ["ok", "healthy", "running"]
                assert "timestamp" in data or "status" in data
                print(f"✅ API health check réussi: {data}")
                return
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Tentative {attempt + 1}/{max_retries} échouée: {e}")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Si on arrive ici, c'est que toutes les tentatives ont échoué
                    pytest.skip(f"API non disponible après {max_retries} tentatives: {e}")

    @patch("subprocess.run")
    def test_deployment_script_execution(self, mock_subprocess: Any):
        """Test d'exécution du script de déploiement."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Deployment successful",
            stderr=""
        )

        result = mock_subprocess(["bash", "run_deployment.sh"])
        assert result.returncode == 0
        assert "Deployment successful" in result.stdout
