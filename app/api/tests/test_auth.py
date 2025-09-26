import pytest
from fastapi.testclient import TestClient
from datetime import timedelta
import jwt

from app.api.main import app
from app.api.security.auth import auth_manager
from app.api.config.settings import settings

client = TestClient(app)


class TestAuthentication:
    """Tests pour l'authentification"""

    def test_create_access_token(self):
        data = {"sub": "test_user"}
        token = auth_manager.create_access_token(data)
        assert isinstance(token, str)
        assert len(token) > 0

        # Vérifier que le token peut être décodé
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        assert payload["sub"] == "test_user"
        assert "exp" in payload

    def test_verify_token_valid(self):
        data = {"sub": "test_user"}
        token = auth_manager.create_access_token(data)
        payload = auth_manager.verify_token(token)
        assert payload["sub"] == "test_user"

    def test_verify_token_expired(self):
        data = {"sub": "test_user"}
        expires_delta = timedelta(seconds=-1)  # Token expiré
        token = auth_manager.create_access_token(data, expires_delta)
        with pytest.raises(Exception):  # JWT error expected
            auth_manager.verify_token(token)

    def test_verify_token_invalid(self):
        invalid_token = "invalid.token.here"
        with pytest.raises(Exception):
            auth_manager.verify_token(invalid_token)

    def test_password_hashing(self):
        password = "test_password_123"
        long_password = "A" * 100  # 100 caractères
        safe_long_password = long_password[:72]

        # Hasher le mot de passe standard
        hashed = auth_manager.get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 0

        # Vérifier le mot de passe standard
        assert auth_manager.verify_password(password, hashed)
        assert not auth_manager.verify_password("wrong_password", hashed)

        # Hasher et vérifier le mot de passe limite (72 caractères)
        hashed_long = auth_manager.get_password_hash(long_password)
        assert hashed_long != safe_long_password
        assert len(hashed_long) > 0
        assert auth_manager.verify_password(long_password, hashed_long)
        assert not auth_manager.verify_password("wrong_password", hashed_long)


class TestAPIKeyAuthentication:
    """Tests pour l'authentification par clé API"""

    def test_valid_api_key(self):
        headers = {"X-API-Key": "test_api_key"}
        response = client.get("/api/v1/", headers=headers)
        assert response.status_code == 200

    def test_invalid_api_key(self):
        headers = {"X-API-Key": "invalid_api_key"}
        response = client.get("/api/v1/", headers=headers)
        assert response.status_code == 401

    def test_missing_api_key(self):
        response = client.get("/api/v1/")
        assert response.status_code == 401