# path: mlops/6_Regression/1_Encoding/encoding_api.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from subprocess import run, CalledProcessError, CompletedProcess

app = FastAPI(title="Encoding API", version="1.0.0")

class EncodingParams(BaseModel):
    data_path: str = "data/df_cluster.csv"
    output_path: str = "data/encoded"

@app.post("/run")
def run_encoding(params: EncodingParams):
    try:
        cp: CompletedProcess = run([
            "python", "mlops/6_Regression/1_Encoding/encoding.py",
            "--data-path", params.data_path,
            "--output", params.output_path
        ])
        return {"status": "ok" if cp.returncode == 0 else "error", "return_code": cp.returncode}
    except CalledProcessError as e:
        return {"status": "error", "return_code": e.returncode, "message": str(e)}

