#seasonal_decomp
from fastapi import FastAPI
from pydantic import BaseModel
from subprocess import run

app = FastAPI()

class seasonaldecompParams(BaseModel):
    input_path: str
    output_path: str

@app.post("/run")
def run_seasonaldecomp(params: seasonaldecompParams):
    result = run([
        "python", "mlops/Serie_temporelle/seasonal_decomp.py",
        "--input-path", params.input_path,
        "--output-path", params.output_path
    ])
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode
    }

