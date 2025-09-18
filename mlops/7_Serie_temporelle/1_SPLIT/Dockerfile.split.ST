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

COPY ../../exports /app/exports
COPY mlops/7_Serie_temporelle/1_SPLIT/requirements.txt /app/mlops/7_Serie_temporelle/1_SPLIT/requirements.txt
COPY mlops/7_Serie_temporelle/1_SPLIT /app/mlops/7_Serie_temporelle/1_SPLIT

COPY ../../data /app/data



RUN pip install --upgrade pip && \
    pip install -r /app/mlops/7_Serie_temporelle/1_SPLIT/requirements.txt
    

ENTRYPOINT ["uvicorn", "split_ST_api:app", "--host", "0.0.0.0", "--port", "8006"]
