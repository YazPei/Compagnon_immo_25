from fastapi import FastAPI
from pydantic import BaseModel
from mlops.preprocessing.preprocessing import run_preprocessing_pipeline

app = FastAPI(title="Preprocessing API", version="1.0.0")

class PreprocessingParams(BaseModel):
    input_path: str
    output_folder1: str
    output_folder2: str

@app.post("/run")
def run_preprocessing(params: PreprocessingParams):
    try:
        run_preprocessing_pipeline(
            input_path=params.input_path,
            output_path=params.output_folder1  # si tu utilises juste 1 output
        )
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

