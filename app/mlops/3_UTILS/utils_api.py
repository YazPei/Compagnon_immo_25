# utils
from fastapi import FastAPI
from pydantic import BaseModel
from subprocess import run

app = FastAPI()


class utilsParams(BaseModel):
    input_path: str
    output_path: str

@app.post("/run")
def run_utils(params: utilsParams):
    result = run([
        "python", "mlops/Regression/utils.py",
        "--input-path", params.input_path,
        "--output-path", params.output_path
    ])
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode
    }

