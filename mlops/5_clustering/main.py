# main.py (conteneur clustering)
from fastapi import FastAPI
from pydantic import BaseModel
from mlops.5_clustering.Clustering import run_clustering_pipeline 

app = FastAPI(title="Clustering Step", version="1.0.0")

class ClusteringRequest(BaseModel):
    input_path: str
    output_path1: str
    output_path2: str

@app.post("/run")
def run_step(req: ClusteringRequest):
    try:
        run_clustering_pipeline(req.input_path, req.output_path)
        return {"status": "success", "message": "✅ Clustering terminé"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

