FROM python:3.11-slim-bookworm
WORKDIR /app

COPY mlops/Serie_temporelle/load_split.py /app/mlops/Serie_temporelle/
COPY services /app

ENTRYPOINT ["uvicorn", "split_ST_api:app", "--host", "0.0.0.0", "--port", "8006"]
