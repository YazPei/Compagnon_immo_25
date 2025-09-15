FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
    
# 1. Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY mlops/7_Serie_temporelle/3_SARIMAX /app/7_Serie_temporelle/3_SARIMAX
COPY data/ /app/data/
COPY exports/ /app/exports


RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    
EXPOSE 8007
ENTRYPOINT ["uvicorn", "sarimax_api:app", "--host", "0.0.0.0", "--port", "8007"]

