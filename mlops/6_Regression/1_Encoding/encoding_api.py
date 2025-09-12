#encoding
from fastapi import FastAPI
from pydantic import BaseModel
from subprocess import run

app = FastAPI()

class EncodingParams(BaseModel):
    input_path: str
    output_path: str

@app.post("/run")
def run_encoding(params: EncodingParams):
    result = run([
        "python", "mlops/Regression/encoding.py",
        "--input-path", params.input_path,
        "--output-path", params.output_path
    ])
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode
    }

