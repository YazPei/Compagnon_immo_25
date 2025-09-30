"""
Tests d'intégration pour vérifier le fonctionnement global des pipelines.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


@pytest.mark.integration
def test_health_check():
    """Vérifie que l'endpoint de santé fonctionne correctement."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    # Accepte aussi "ok" comme valeur valide
    assert data["status"] in ["healthy", "degraded", "unhealthy", "ok"]


@pytest.mark.integration
def test_estimation_endpoint():
    """Vérifie que l'endpoint d'estimation retourne une réponse valide."""
    payload = {"surface": 100, "nb_pieces": 4, "code_postal": "75001"}
    api_key = os.getenv("API_SECRET_KEY", "yasmineketsia")
    headers = {"X-API-Key": api_key}
    response = client.post("/api/v1/estimation", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "estimation" in data
    assert "prix" in data["estimation"]
    assert data["estimation"]["prix"] > 0
