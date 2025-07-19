FROM compagnon_immo-sarimax

WORKDIR /app


COPY mlops/Serie_temporelle /app/mlops/Serie_temporelle
COPY api_rest /app


ENTRYPOINT ["uvicorn", "evaluate_ST_api:app", "--host", "0.0.0.0", "--port", "8008"]


