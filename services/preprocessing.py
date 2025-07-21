# api_preprocessing.py
from fastapi import FastAPI, Query
from subprocess import run

app = FastAPI()

@app.post("/run")
def run_preprocessing(input_path: str = Query(...), output_path: str = Query(...)):
    result = run([
        "python", "preprocessing.py",
        "--input-path", input_path,
        "--output-path", output_path
    ])
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode
    }

