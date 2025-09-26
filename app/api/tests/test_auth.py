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
        """Test création d'un token d'accès"""
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
        """Test vérification d'un token valide"""
        data = {"sub": "test_user"}
        token = auth_manager.create_access_token(data)

        payload = auth_manager.verify_token(token)
        assert payload["sub"] == "test_user"

    def test_verify_token_expired(self):
        """Test vérification d'un token expiré"""
        data = {"sub": "test_user"}
        expires_delta = timedelta(seconds=-1)  # Token expiré
        token = auth_manager.create_access_token(data, expires_delta)

        with pytest.raises(Exception):  # JWT error expected
            auth_manager.verify_token(token)

    def test_verify_token_invalid(self):
        """Test vérification d'un token invalide"""
        invalid_token = "invalid.token.here"

        with pytest.raises(Exception):
            auth_manager.verify_token(invalid_token)

    def test_password_hashing(self):
        """Test hashage et vérification de mot de passe"""
        password = "test_password_123"

        # Hasher le mot de passe
        hashed = auth_manager.get_password_hash(password)
        assert hashed is not None

        # Vérifier que le hashage fonctionne
        assert auth_manager.verify_password(password, hashed)

        # Vérifier qu'un mauvais mot de passe est rejeté
        assert not auth_manager.verify_password("wrong_password", hashed)

    def test_password_hashing_long_password(self):
        """Test hashage avec mot de passe > 72 octets pour bcrypt"""
        long_password = "A" * 100  # 100 caractères ASCII, donc 100 octets

        # Hasher le mot de passe
        hashed = auth_manager.get_password_hash(long_password)
        assert hashed is not None

        # Le mot de passe original doit fonctionner même s'il est tronqué en interne
        assert auth_manager.verify_password(long_password, hashed)

        # Vérifier qu'un mauvais mot de passe est rejeté
        assert not auth_manager.verify_password("B" * 100, hashed)

        # Test spécifique: vérifier que seuls les 72 premiers caractères comptent
        modified_password = "A" * 72 + "B" * 28  # Modifie les caractères après la limite de 72
        assert auth_manager.verify_password(modified_password, hashed)

        # Test pour confirmer qu'une modification dans les 72 premiers caractères est bien détectée
        modified_password2 = "B" + "A" * 99  # Modifie le premier caractère
        assert not auth_manager.verify_password(modified_password2, hashed)

class TestAPIKeyAuthentication:
    """Tests pour l'authentification par clé API"""

    def test_valid_api_key(self):
        """Test avec une clé API valide"""
        headers = {"X-API-Key": "test_api_key"}
        response = client.get("/api/v1/", headers=headers)

        # L'endpoint devrait être accessible
        assert response.status_code == 200

    def test_invalid_api_key(self):
        """Test avec une clé API invalide"""
        headers = {"X-API-Key": "invalid_api_key"}
        response = client.get("/api/v1/", headers=headers)

        # L'endpoint ne devrait pas être accessible
        assert response.status_code == 401

    def test_missing_api_key(self):
        """Test sans clé API"""
        response = client.get("/api/v1/")

        # L'endpoint ne devrait pas être accessible
        assert response.status_code == 401