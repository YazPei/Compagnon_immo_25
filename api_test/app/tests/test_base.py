
from fastapi import FastAPI
from api_test.app.routes.routers import historique  
from api_test.app.routes.routers import estimation  

app = FastAPI(title="API d'Estimation Immobilière", version="1.0")

# Intégration des routes et vérification de leur accès
app.include_router(estimation.router, prefix="/api/v1")
app.include_router(historique.router, prefix="/api/v1")

#Ici on vérifie la réponse de l'API avec la méthode GET
@app.get("/", tags=["Base"])
def root():
    return {"message": "Bienvenue sur mon Compagnon d'Estimation Immobilière."}

#Ici on vérifie que l'API tourne correctement avec la méthode GET
@app.get("/health", tags=["Base"])
def health():
    return {"status": "ok"}

#Ici on récupère la version de l'API avec la méthode GET sans passer par les fonctionnalités métier
@app.get("/version", tags=["Base"])
def version():
    return {"version": app.version}
