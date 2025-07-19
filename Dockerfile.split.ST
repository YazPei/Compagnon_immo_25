FROM compagnon_immo-clust
WORKDIR /app

COPY mlops/Serie_temporelle/load_split.py /app/mlops/Serie_temporelle/

ENTRYPOINT ["uvicorn", "split_ST_api:app", "--host", "0.0.0.0", "--port", "8006"]
