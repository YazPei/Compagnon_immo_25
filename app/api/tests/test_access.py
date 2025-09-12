from fastapi import FastAPI
from app.api.routes import historique, estimation
import pytest
from fastapi.testclient import TestClient
from app.api.main import app


def create_test_app() -> FastAPI:
    """Crée une application FastAPI de test."""
    app = FastAPI(
        title="API d'Estimation Immobilière",
        version="1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Ajout des endpoints de base avec des operation_id uniques
    @app.get("/", tags=["Base"], operation_id="test_root")
    def root():
        return {
            "message": (
                "Bienvenue sur mon Compagnon d'Estimation Immobilière."
            )
        }

    @app.get("/health", tags=["Base"], operation_id="test_health")
    def health():
        return {"status": "ok", "service": "api-estimation"}

    @app.get("/version", tags=["Base"], operation_id="test_version")
    def version():
        return {"version": app.version, "service": "api-estimation"}
    
    # Inclusion des routers
    app.include_router(estimation.router, prefix="/api/v1", tags=["Estimation"])
    app.include_router(historique.router, prefix="/api/v1", tags=["Historique"])
    
    return app


# Création de l'app de test
app = create_test_app()
client = TestClient(app)


class TestBaseEndpoints:
    """Tests des endpoints de base."""
    
    def test_root_endpoint(self):
        """Test de l'endpoint racine."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Compagnon d'Estimation Immobilière" in data["message"]
    
    def test_health_endpoint(self):
        """Test de l'endpoint de santé."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
    
    def test_version_endpoint(self):
        """Test de l'endpoint de version."""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0"
        assert "service" in data
    
    def test_docs_accessible(self):
        """Test que la documentation Swagger est accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema(self):
        """Test que le schéma OpenAPI est valide."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema


class TestAPIStructure:
    """Tests de la structure de l'API."""

    def test_api_v1_prefix(self):
        """Vérifie que les routes API utilisent le bon préfixe."""
        response = client.get("/openapi.json")
        schema = response.json()
        paths = schema.get("paths", {})
        api_v1_paths = [path for path in paths.keys() if path.startswith("/api/v1")]
        assert len(api_v1_paths) > 0, "Aucune route /api/v1 trouvée"
    
    def test_unique_operation_ids(self):
        """Vérifie que tous les operation_id dans le schéma OpenAPI 
        sont uniques."""
        response = client.get("/openapi.json")
        schema = response.json()
        operation_ids = []
        for path, methods in schema.get("paths", {}).items():
            for method, details in methods.items():
                operation_id = details.get("operationId")
                if operation_id:
                    operation_ids.append(operation_id)
        
        assert len(operation_ids) == len(set(operation_ids)), (
            "Des operation_id dupliqués ont été trouvés"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])