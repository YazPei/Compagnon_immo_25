FROM compagnon_immo-split-st
WORKDIR /app


COPY mlops/Serie_temporelle/seasonal_decomp.py /app/mlops/Serie_temporelle/
COPY services /app

ENTRYPOINT ["uvicorn", "seasonal_decomp_st_api:app", "--host", "0.0.0.0", "--port", "8007"]

