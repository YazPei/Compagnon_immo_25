#evaluate_ST	
from subprocess import run

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class evaluateParams(BaseModel):
    input_path: str
    output_path: str

@app.post("/run")
def run_evaluate(params: evaluateParams):
    result = run([
        "python", "mlops/Serie_temporelle/evaluate_ST.py",
        "--input-path", params.input_path,
        "--output-path", params.output_path
    ])
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode
    }
/home/yazpei/projets/compagnon_immo/MLE/compagnon_immo/Compagnon_immo/mlops/Serie_temporelle
