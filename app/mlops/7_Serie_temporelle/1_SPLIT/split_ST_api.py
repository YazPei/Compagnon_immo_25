# load_split
from subprocess import run

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class splitstParams(BaseModel):
    input_path: str
    output_path: str


@app.post("/run")
def run_splitst(params: splitstParams):
    result = run(
        [
            "python",
            "mlops/Serie_temporelle/load_split.py",
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
