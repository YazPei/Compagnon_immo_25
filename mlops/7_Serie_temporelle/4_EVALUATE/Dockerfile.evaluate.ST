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

COPY mlops/7_Serie_temporelle/4_EVALUATE/requirements.txt /app/mlops/4_EVALUATE/requirements.txt
COPY mlops/7_Serie_temporelle/4_EVALUATE /app/mlops/4_EVALUATE
COPY ../../outputs /app/outputs
COPY ../../data /app/data


RUN pip install --upgrade pip && \
    pip install -r /app/mlops/4_EVALUATE/requirements.txt
    
EXPOSE 8008
ENTRYPOINT ["uvicorn", "evaluate_ST_api:app", "--host", "0.0.0.0", "--port", "8008"]


