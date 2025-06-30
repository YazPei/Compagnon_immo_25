from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_test.api import estimation
from api_test.api import historique

app = FastAPI(
    title="API Compagnon Immobilier",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Les endpoints de base
@app.get("/", tags=["Base"])
def root():
    return {"message": "Bienvenue sur l'API Compagnon Immobilier."}

@app.get("/health", tags=["Base"])
def health():
    return {"status": "ok"}

@app.get("/version", tags=["Base"])
def version():
    return {"version": app.version}

# Inclusion des routes métiers
app.include_router(estimation.router, prefix="/api/v1", tags=["Estimation"])
app.include_router(historique.router, prefix="/api/v1", tags=["Historique"])

