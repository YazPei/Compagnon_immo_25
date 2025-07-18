#preprocessing	
from fastapi import FastAPI
from pydantic import BaseModel
from subprocess import run

app = FastAPI()

class PreprocessingParams(BaseModel):
    input_path: str
    output_folder1: str
    output_folder2: str
@app.post("/run")
def run_preprocessing(params: PreprocessingParams):
    result = run([
        "python", "mlops/preprocessing/preprocessing.py",
        "--input-path", params.input_path,
        "--output-folder1", params.output_folder1,
        "--output-folder2", params.output_folder2
    ])
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode
    }

