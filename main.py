# main.py
from fastapi import FastAPI, UploadFile, File
from subprocess import run
from pydantic import BaseModel
import shutil
import os

app = FastAPI(title="Pipeline API")

# === Endpoints de chaque étape ===

@app.post("/preprocess")
def preprocess(input_path: str, output_path: str):
    result = run([
        "python", "preprocessing.py",
        "--input-path", input_path,
        "--output-path", output_path
    ])
    return {"status": "done", "return_code": result.returncode}

@app.post("/cluster")
def cluster(input_path: str, output_path: str):
    result = run([
        "python", "Clustering.py",
        "--input-path", input_path,
        "--output-path", output_path
    ])
    return {"status": "done", "return_code": result.returncode}

@app.post("/encode")
def encode(data_path: str, output: str, target: str = "prix_m2_vente"):
    result = run([
        "python", "encoding.py",
        "--data-path", data_path,
        "--output", output,
        "--target", target
    ])
    return {"status": "done", "return_code": result.returncode}

@app.post("/train-lgbm")
def train(encoded_folder: str):
    result = run([
        "python", "train_lgbm.py",
        "--encoded-folder", encoded_folder
    ])
    return {"status": "done", "return_code": result.returncode}

