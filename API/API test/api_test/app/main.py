from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_test.app.api import estimation, historique
import os
from api_test.app.security.rate_limit import RateLimitMiddleware

app = FastAPI(title="API d'Estimation Immobilière", version="1.0")

# Rate limiting
app.add_middleware(RateLimitMiddleware)

# CORS (pour tests locaux)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(estimation.router, prefix="/api/v1")
app.include_router(historique.router, prefix="/api/v1")
