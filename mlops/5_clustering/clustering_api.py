# path: mlops/5_clustering/Clustering_api.py
from __future__ import annotations
import importlib.util
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# charge Clustering.py par chemin (le dossier '5_clustering' n'est pas importable)
CLUSTERING_PATH = Path(__file__).with_name("Clustering.py")
spec = importlib.util.spec_from_file_location("clustering_mod", CLUSTERING_PATH)
clustering_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(clustering_mod)  # type: ignore

app = FastAPI(title="Clustering API", version="1.0.0")

class ClusteringRequest(BaseModel):
    input_path: str
    output_path: str

@app.post("/run")
def run_step(req: ClusteringRequest):
    try:
        clustering_mod.run_clustering_pipeline(req.input_path, req.output_path)
        return {"status": "success", "message": "✅ Clustering terminé"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

