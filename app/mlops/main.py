# main.py
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Orchestrateur Compagnon Immo")


class PipelineRequest(BaseModel):
    input_path: str
    output_path: str
    target: str = "prix_m2_vente"

@app.post("/run_full_pipeline")
def run_pipeline(req: PipelineRequest):
    steps = [
        {
            "name": "Preprocessing",
            "url": "http://preprocessing:8001/run",
            "params": {"input_path": req.input_path, "output_path": req.output_path}
        },
        {
            "name": "Clustering",
            "url": "http://clustering:8002/run",
            "params": {"input_path": req.output_path, "output_path": req.output_path}
        },
        {
            "name": "Encoding",
            "url": "http://encoding:8003/run",
            "params": {"data_path": f"{req.output_path}/df_sales_clustered.csv", "output": req.output_path, "target": req.target}
        },
        {
            "name": "utils evaluate LGBM",
            "url": "http://train_lgbm:8009/run",
            "params": {"input_path": req.input_path}
        },        
        {
            "name": "Train LGBM",
            "url": "http://train_lgbm:8004/run",
            "params": {"encoded_folder": req.output_path}
        },
        {
            "name": "Analyse",
            "url": "http://train_lgbm:8005/run",
            "params": {"model": req.output_path}
        },        
        {
            "name": "Split Time Series",
            "url": "http://train_lgbm:8006/run",
            "params": {"input_path": req.input_path, "output_path": req.output_path}
        },
        {
            "name": "decompose",
            "url": "http://train_lgbm:8007/decompose",
            "params": {"input_path": req.input_path, "output_path": req.output_path}
        },
        {
            "name": "sarimax",
            "url": "http://train_lgbm:8007/sarimax",
            "params": {"input_path": req.input_path, "output_path": req.output_path}
        },        
        {
            "name": "evaluate",
            "url": "http://train_lgbm:8008/run",
            "params": {"input_path": req.input_path, "output_path": req.output_path}
        },        
    ]

    results = []
    for step in steps:
        try:
            r = requests.post(step["url"], params=step["params"])
            results.append({"step": step["name"], "status": "OK" if r.status_code == 200 else "❌ Failed", "response": r.json()})
        except Exception as e:
            results.append({"step": step["name"], "status": "Error", "error": str(e)})

    return {"pipeline_status": "done", "results": results}


