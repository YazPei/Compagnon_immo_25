FROM python:3.10-slim

WORKDIR /app


COPY mlops/Serie_temporelle /app/mlops/Serie_temporelle
COPY services /app

EXPOSE 8008
ENTRYPOINT ["uvicorn", "evaluate_ST_api:app", "--host", "0.0.0.0", "--port", "8008"]


