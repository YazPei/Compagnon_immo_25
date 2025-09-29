# Dockerfile API - Compagnon Immo
# Multi-stage pour une image légère et sécurisée

FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

# Installer les dépendances en amont pour profiter du cache
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn dvc apache-airflow==2.8.3 psycopg2-binary

RUN pip check  # Vérifie les conflits de dépendances
RUN chmod -R 755 /app  # Assure les permissions correctes

# Copier uniquement ce qui est nécessaire à l'API
COPY app ./app
COPY params.yaml ./params.yaml
COPY .dvc ./.dvc

# Utilisateur non-root
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Variables par défaut (peuvent être surchargées par Compose/ENV)
ENV ENVIRONMENT=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    WORKERS=4 \
    LOG_LEVEL=INFO

# Commande par défaut (overridable par docker-compose)
CMD [ \
    "gunicorn", "app.api.main:app", \
    "-w", "4", \
    "-k", "uvicorn.workers.UvicornWorker", \
    "-b", "0.0.0.0:8000" \
    ]

