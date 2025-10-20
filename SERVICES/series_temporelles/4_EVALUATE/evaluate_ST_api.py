# SERVICES/series_temporelles/4_EVALUATE/evaluate_ST_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from subprocess import run

app = FastAPI()

class evaluateParams(BaseModel):
    input_path: str
    output_path: str

@app.post("/run")
def run_evaluate(params: evaluateParams):
    result = run([
        "python", "SERVICES/series_temporelles/4_EVALUATE/evaluate_ST.py",
        "--input-path", params.input_path,
        "--output-path", params.output_path
    ])
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode
    }
