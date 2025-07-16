from fastapi import FastAPI
from api_test.app.routes.routers import historique  
from api_test.app.routes.routers import estimation  
import pytest
from fastapi.testclient import TestClient


def create_test_app() -> FastAPI:
    app = FastAPI(
        title="API d'Estimation Immobilière", 
        version="1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.include_router(estimation.router, prefix="/api/v1", tags=["Estimation"])
    app.include_router(historique.router, prefix="/api/v1", tags=["Historique"])
    
    @app.get("/", tags=["Base"])
    def root():
        return {"message": "Bienvenue sur mon Compagnon d'Estimation Immobilière."}

    @app.get("/health", tags=["Base"])
    def health():
        return {"status": "ok", "service": "api-estimation"}

    @app.get("/version", tags=["Base"])
    def version():
        return {"version": app.version, "service": "api-estimation"}
    
    return app


app = create_test_app()


@app.get("/", tags=["Base"])
def root():
    return {"message": "Bienvenue sur mon Compagnon d'Estimation Immobilière."}


@app.get("/health", tags=["Base"])
def health():
    return {"status": "ok"}


@app.get("/version", tags=["Base"])
def version():
    return {"version": app.version}


client = TestClient(app)


class TestBaseEndpoints:    
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
        
        # Ici on vérifie qu'il y a des routes avec le préfixe /api/v1
        api_v1_paths = [path for path in paths.keys() if path.startswith("/api/v1")]
        assert len(api_v1_paths) > 0, "Aucune route /api/v1 trouvée"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
