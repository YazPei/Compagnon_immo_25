from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import make_asgi_app
import uvicorn
import logging

from app.api.config.settings import settings
from app.api.utils.exception_handlers import (
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from app.api.middleware.security import security_headers_middleware, request_logging_middleware
from app.api.security.rate_limit import rate_limit_middleware
from app.api.routes import (
    estimation,
    historique,
    health_routes,
    dvc_routes,
    deployment
)

# Configuration du logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialisation de l'application FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API pour le projet Compagnon Immobilier, offrant des services d'estimation, historique, gestion des modèles et monitoring.",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc"
)

# Middleware de sécurité
app.middleware("http")(security_headers_middleware)
app.middleware("http")(request_logging_middleware)
app.middleware("http")(rate_limit_middleware)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # Streamlit
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Middleware Trusted Hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
)

# Gestion des exceptions
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(ValueError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Inclusion des routers
api_router = FastAPI()

api_router.include_router(
    health_routes.router,
    prefix="/health",
    tags=["Health"]
)

api_router.include_router(
    estimation.router,
    prefix="/estimation",
    tags=["Estimation"]
)

api_router.include_router(
    historique.router,
    prefix="/historique",
    tags=["Historique"]
)

api_router.include_router(
    dvc_routes.router,
    prefix="/dvc",
    tags=["DVC"]
)

api_router.include_router(
    deployment.router,
    prefix="/deployment",
    tags=["Deployment"]
)

app.include_router(api_router, prefix=settings.API_V1_STR)

# Exposition des métriques Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Endpoints de base
@app.get("/", tags=["Base"], operation_id="root_endpoint")
def root():
    """Point d'entrée principal de l'API."""
    return {"message": "Bienvenue sur l'API Compagnon Immobilier."}

@app.get("/health", tags=["Base"], operation_id="health_check")
def health():
    """Vérification de l'état de santé de l'API."""
    return {"status": "ok", "service": "api-compagnon-immobilier"}

@app.get("/version", tags=["Base"], operation_id="get_version")
def version():
    """Retourne la version de l'API."""
    return {"version": app.version, "service": "api-compagnon-immobilier"}

@app.get("/status", tags=["Base"], operation_id="api_status")
async def api_status():
    """Statut global de l'API."""
    return {
        "status": "operational",
        "services": ["estimation", "historique", "dvc", "deployment"]
    }

# Endpoint pour effectuer des prédictions
@app.post("/predict", tags=["ML"], operation_id="predict")
async def predict(features: dict):
    """Effectuer une prédiction avec les modèles ML."""
    try:
        result = ml_service.predict(features)
        return {
            "status": "success",
            "prediction": result
        }
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur prédiction: {str(e)}"
        )

# Endpoint pour récupérer les informations des modèles
@app.get("/models", tags=["ML"], operation_id="get_models_info")
async def get_models_info():
    """Retourne les informations sur les modèles ML chargés."""
    return ml_service.get_models_info()

# Lancement de l'application
if __name__ == "__main__":
    uvicorn.run(
        "app.api.routes.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )