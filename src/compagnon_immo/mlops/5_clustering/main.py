# path: mlops/5_clustering/main.py
from __future__ import annotations
import uvicorn
from .Clustering_api import app  # même dossier, import relatif

if __name__ == "__main__":
    uvicorn.run("mlops.5_clustering.main:app", host="0.0.0.0", port=8002, reload=False)

