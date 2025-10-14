# analyse
from subprocess import run

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class analyseParams(BaseModel):
    input_path: str
    output_path: str


@app.post("/run")
def run_analyse(params: analyseParams):
    result = run(
        [
            "python",
            "mlops/Regression/analyse.py",
            "--input-path",
            params.input_path,
            "--output-path",
            params.output_path,
        ]
    )
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "return_code": result.returncode,
    }
