FROM compagnon_immo-decomp
WORKDIR /app

COPY mlops/Serie_temporelle/sarimax_train.py /app/mlops/Serie_temporelle/	
COPY data/ /app/data/
COPY exports/ /app/exports

EXPOSE 8007
ENTRYPOINT ["uvicorn", "sarimax_api:app", "--host", "0.0.0.0", "--port", "8007"]

