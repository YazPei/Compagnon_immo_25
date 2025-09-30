# main.py (conteneur preprocessing)
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from mlops.preprocessing_4.preprocessing import run_preprocessing_pipeline

app = FastAPI(title="Preprocessing Step", version="1.0.0")


class PreprocessingRequest(BaseModel):
    input_path: str
    output_path: str


@app.post("/run")
def run_step(req: PreprocessingRequest):
    try:
        run_preprocessing_pipeline(req.input_path, req.output_path)
        return {"status": "success", "message": "✅ Preprocessing terminé"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        "mlops.4_preprocessing.main:app", host="0.0.0.0", port=8002, reload=False
    )
