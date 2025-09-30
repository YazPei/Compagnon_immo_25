import subprocess

from fastapi import FastAPI

app = FastAPI()


@app.get("/run-dvc")
def run_dvc():
    try:
        result = subprocess.run(
            ["bash", "run_dvc.sh"], capture_output=True, text=True, check=True
        )
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "output": e.stderr}
