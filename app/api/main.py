# app/api/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.config.settings import settings

from app.api.routes import main as main_routes
from app.api.routes import estimation as estimation_routes
from app.api.routes import health as health_routes
from app.api.routes import historique as historique_routes
from app.api.routes import metrics as metrics_routes

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API d'estimation immobilière",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints généraux (non versionnés)
@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
    }

# Routers (attention aux préfixes attendus par les tests)
app.include_router(health_routes.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(main_routes.router,   prefix="/api/v1",        tags=["Main"])
app.include_router(estimation_routes.router, prefix="/api/v1",    tags=["Estimation"])
app.include_router(historique_routes.router, prefix="/api/v1/historique", tags=["Historique"])

# métriques (optionnel)
app.include_router(metrics_routes.router, tags=["Monitoring"])

# Probes K8s (facultatif pour tests)
@app.get("/liveness")
async def liveness():
    return {"status": "alive"}

@app.get("/readiness")
async def readiness():
    return {"status": "ready"}

